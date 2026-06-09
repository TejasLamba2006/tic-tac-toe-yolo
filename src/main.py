from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .ai.move_selector import MoveDecision, recommend_move
from .ai.yolo_inference import YoloInference
from .vision.board_detector import BoardDetectionResult, BoardDetector
from .vision.board_state import BoardObservation, BoardStateEstimator, Detection, format_board
from .vision.camera import configure_capture, open_camera, parse_camera_source
from .vision.perspective import PerspectiveTransform, canonical_cell_polygon, warp_image, warp_points
from .vision.stability import BoardGeometryTracker


@dataclass
class AppConfig:
    camera: str
    weights: str
    ai_color: str
    board_size: int
    frame_width: int
    frame_height: int
    fps: int
    confidence_threshold: float
    board_min_confidence: float
    iou_threshold: float
    image_size: int
    smoothing_window: int
    auto_max_index: int
    headless: bool
    device: str | None
    npu: bool


@dataclass(frozen=True)
class FrameAnalysis:
    frame: np.ndarray
    warped_frame: np.ndarray
    board_result: BoardDetectionResult
    transform: PerspectiveTransform
    observation: BoardObservation
    decision: MoveDecision
    inference_ms: float
    model_run_ms: float | None
    move_ms: float
    analysis_ms: float
    fps: float
    detections: tuple[Detection, ...]


def _label_color(label: str, class_id: int) -> tuple[int, int, int]:
    palette = {
        "empty": (140, 140, 140),
        "red_ball": (0, 0, 255),
        "yellow_ball": (0, 220, 255),
    }
    if label in palette:
        return palette[label]

    seed = (class_id * 73) % 255
    return (seed, 255 - seed, 160 + seed // 2)


def _draw_grid(image: np.ndarray) -> np.ndarray:
    grid = image.copy()
    height, width = grid.shape[:2]
    step_x = width // 3
    step_y = height // 3

    for index in (1, 2):
        cv2.line(grid, (index * step_x, 0), (index * step_x, height),
                 (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(grid, (0, index * step_y), (width, index * step_y),
                 (80, 80, 80), 1, cv2.LINE_AA)

    return grid


def _draw_detection_boxes(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        color = _label_color(detection.label, detection.class_id)
        left = int(np.clip(x1, 0, width - 1))
        top = int(np.clip(y1, 0, height - 1))
        right = int(np.clip(x2, 0, width - 1))
        bottom = int(np.clip(y2, 0, height - 1))

        cv2.rectangle(annotated, (left, top),
                      (right, bottom), color, 2, cv2.LINE_AA)
        label = f"{detection.label} {detection.confidence:.2f}"
        text_y = max(14, top - 6)
        cv2.putText(annotated, label, (left, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    return annotated


def _overlay_inset(frame: np.ndarray, inset: np.ndarray, x: int, y: int) -> np.ndarray:
    rendered = frame.copy()
    inset_height, inset_width = inset.shape[:2]
    frame_height, frame_width = rendered.shape[:2]
    x = max(0, min(frame_width - inset_width - 8, x))
    y = max(0, min(frame_height - inset_height - 8, y))

    roi = rendered[y:y + inset_height, x:x + inset_width]
    if roi.shape[:2] != inset.shape[:2]:
        return rendered

    blended = cv2.addWeighted(roi, 0.10, inset, 0.90, 0)
    rendered[y:y + inset_height, x:x + inset_width] = blended
    cv2.rectangle(rendered, (x, y), (x + inset_width, y +
                  inset_height), (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


def _build_board_inset(analysis: FrameAnalysis, inset_size: int = 280) -> np.ndarray:
    board_preview = analysis.warped_frame.copy()
    board_preview = _draw_grid(board_preview)
    board_preview = _draw_detection_boxes(board_preview, analysis.detections)
    resized = cv2.resize(
        board_preview, (inset_size, inset_size), interpolation=cv2.INTER_AREA)

    cv2.rectangle(resized, (0, 0), (inset_size - 1, 24), (18, 18, 18), -1)
    cv2.putText(resized, f"Warped board | {analysis.board_result.method}", (
        8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return resized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tic-tac-toe vision pipeline")
    parser.add_argument("--gui", default="opencv", choices=("opencv", "tkinter"),
                        help="User interface mode")
    parser.add_argument("--camera", default="auto",
                        help="Camera source: index, /dev/videoX, or auto")
    parser.add_argument(
        "--weights",
        default="runs/detect/train/weights/best.pt",
        help="YOLO weights path (.pt, .onnx, or .tflite)",
    )
    parser.add_argument("--ai-color", default="Y",
                        choices=("R", "Y"), help="Which player the AI controls")
    parser.add_argument("--board-size", type=int, default=320,
                        help="Canonical warped board size in pixels")
    parser.add_argument("--frame-width", type=int,
                        default=1280, help="Requested capture width")
    parser.add_argument("--frame-height", type=int,
                        default=720, help="Requested capture height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Requested camera frame rate")
    parser.add_argument("--confidence-threshold", type=float,
                        default=0.70, help="YOLO confidence threshold")
    parser.add_argument("--board-min-confidence", type=float,
                        default=0.50, help="Minimum confidence required for a cell to stay in the matrix")
    parser.add_argument("--iou-threshold", type=float,
                        default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--image-size", type=int, default=320,
                        help="YOLO inference image size")
    parser.add_argument("--smoothing-window", type=int,
                        default=5, help="Board-state history window used for stabilization")
    parser.add_argument("--auto-max-index", type=int, default=10,
                        help="Highest integer camera index to probe")
    parser.add_argument("--device", default=None,
                        help="Ultralytics device string, for example cpu or 0")
    parser.add_argument("--headless", action="store_true",
                        help="Disable the preview window")
    parser.add_argument("--npu", action="store_true",
                        help="Enable hardware NPU acceleration via vx delegate (STM32MP257)")
    return parser


def analyze_frame(
    frame: np.ndarray,
    detector: YoloInference,
    board_detector: BoardDetector,
    board_estimator: BoardStateEstimator,
    ai_color: str,
    board_size: int,
    geometry_tracker: BoardGeometryTracker | None = None,
    # Fix 2: mutable default acts as a persistent cache cell across calls.
    # Index 0 = last board signature, index 1 = last MoveDecision.
    _cache: list = [None, None],
) -> FrameAnalysis:
    board_result = board_detector.detect(frame)
    if geometry_tracker is not None:
        board_result = geometry_tracker.update(board_result)

    transform = board_result.build_transform(board_size)
    warped_frame = warp_image(frame, transform)

    analysis_start = time.perf_counter()
    infer_start = time.perf_counter()
    # cv2.imwrite("debug_warped.jpg", warped_frame)
    detections = detector.predict(warped_frame)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0
    # print("\nRAW DETECTIONS")

    # for d in detections:
    #     print(
    #         f"{d.label:12} "
    #         f"{d.confidence:.2f} "
    #         f"{d.xyxy}"
    #     )
    observation = board_estimator.estimate(warped_frame, detections)
    move_start = time.perf_counter()
    # Fix 2: skip minimax when the board hasn't changed since the last frame.
    board_signature = tuple(tuple(row) for row in observation.board)
    if board_signature != _cache[0]:
        _cache[0] = board_signature
        _cache[1] = recommend_move(observation.board, ai_player=ai_color)
    decision: MoveDecision = _cache[1]  # type: ignore[assignment]
    move_ms = (time.perf_counter() - move_start) * 1000.0
    analysis_ms = (time.perf_counter() - analysis_start) * 1000.0
    fps = 1000.0 / analysis_ms if analysis_ms > 0 else 0.0

    return FrameAnalysis(
        frame=frame,
        warped_frame=warped_frame,
        board_result=board_result,
        transform=transform,
        observation=observation,
        decision=decision,
        inference_ms=infer_ms,
        model_run_ms=getattr(detector, "last_model_run_ms", None),
        move_ms=move_ms,
        analysis_ms=analysis_ms,
        fps=fps,
        detections=tuple(detections),
    )


def _draw_panel(frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
    rendered = frame.copy()
    height, width = rendered.shape[:2]

    panel_width = min(400, max(280, width // 3))
    panel_height = min(height - 24, max(340, height // 2))
    overlay = rendered.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_width,
                  12 + panel_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.55, rendered, 0.45, 0, rendered)

    y = 40
    line_height = 28
    cv2.putText(rendered, "Current Board", (28, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_height
    for row in analysis.observation.board:
        text = " | ".join(row)
        cv2.putText(rendered, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 220, 255), 2, cv2.LINE_AA)
        y += line_height

    y += 8
    cv2.putText(rendered, f"Model: {analysis.board_result.method}", (
        28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_height
    model_ms = analysis.model_run_ms
    model_time = f"{model_ms:.1f} ms" if model_ms is not None else "n/a"
    cv2.putText(rendered, f"Predict: {analysis.inference_ms:.1f} ms", (
        28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_height
    cv2.putText(rendered, f"Model run: {model_time}", (
        28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_height
    cv2.putText(rendered, f"Move: {analysis.move_ms:.1f} ms", (
        28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_height
    cv2.putText(rendered, f"Frame AI: {analysis.analysis_ms:.1f} ms", (
        28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_height
    cv2.putText(rendered, f"Detections: {len(analysis.detections)}", (
        28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_height

    if analysis.decision.recommendation is not None:
        move = analysis.decision.recommendation
        move_lines = [
            "Best Move",
            f"row={move.row}",
            f"col={move.col}",
        ]
    else:
        move_lines = ["Best Move", analysis.decision.message]

    for text in move_lines:
        cv2.putText(rendered, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 190, 90), 2, cv2.LINE_AA)
        y += line_height

    if analysis.decision.recommendation is not None:
        polygon = canonical_cell_polygon(
            analysis.decision.recommendation.row,
            analysis.decision.recommendation.col,
            analysis.transform.size[0],
        )
        projected = warp_points(polygon, analysis.transform.inverse_matrix)
        cv2.polylines(rendered, [projected.astype(
            np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)

    if not analysis.board_result.fallback:
        cv2.polylines(rendered, [analysis.board_result.corners.astype(
            np.int32)], True, (0, 160, 255), 2, cv2.LINE_AA)

    inset = _build_board_inset(analysis)
    rendered = _overlay_inset(
        rendered, inset, rendered.shape[1] - inset.shape[1] - 12, rendered.shape[0] - inset.shape[0] - 12)

    return rendered


def run_app(config: AppConfig) -> int:
    camera_source = parse_camera_source(config.camera)
    session.release()
    session = open_camera(camera_source, auto_max_index=config.auto_max_index)
    configure_capture(session.capture, config.frame_width,
                      config.frame_height, config.fps)

    yolo = YoloInference(
        weights_path=config.weights,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        image_size=config.image_size,
        device=config.device,
        use_npu=config.npu,
    )
    board_detector = BoardDetector()
    board_estimator = BoardStateEstimator(
        minimum_confidence=config.board_min_confidence,
        smoothing_window=config.smoothing_window)
    geometry_tracker = BoardGeometryTracker()

    window_name = "Tic-Tac-Toe Vision"
    display_enabled = not config.headless
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_signature: tuple[tuple[str, ...], ...] | None = None

    try:
        while True:
            for i in range(10):
                ret, frame = session.read()
                print(
                    f"read {i}: ret={ret}, "
                    f"frame={'None' if frame is None else frame.shape}"
                )

                if ret and frame is not None:
                    break
            ok, frame = session.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read a frame from the camera")

            analysis = analyze_frame(
                frame=frame,
                detector=yolo,
                board_detector=board_detector,
                board_estimator=board_estimator,
                ai_color=config.ai_color,
                board_size=config.board_size,
                geometry_tracker=geometry_tracker,
            )

            rendered = _draw_panel(frame, analysis)
            signature = tuple(tuple(row) for row in analysis.observation.board)
            if signature != last_signature:
                last_signature = signature
                print(format_board(analysis.observation.board))
                print(analysis.decision.message)

            if display_enabled:
                cv2.imshow(window_name, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        session.release()
        if display_enabled:
            cv2.destroyAllWindows()

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig(
        camera=args.camera,
        weights=args.weights,
        ai_color=args.ai_color,
        board_size=args.board_size,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        fps=args.fps,
        confidence_threshold=args.confidence_threshold,
        board_min_confidence=args.board_min_confidence,
        iou_threshold=args.iou_threshold,
        image_size=args.image_size,
        smoothing_window=args.smoothing_window,
        auto_max_index=args.auto_max_index,
        headless=args.headless,
        device=args.device,
        npu=args.npu,
    )
    if args.gui == "tkinter":
        from src.gui.tkinter_app import run_tkinter_app

        return run_tkinter_app(config)

    return run_app(config)


if __name__ == "__main__":
    raise SystemExit(main())
