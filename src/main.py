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
from .vision.board_state import BoardObservation, BoardStateEstimator, format_board
from .vision.camera import configure_capture, open_camera, parse_camera_source
from .vision.perspective import PerspectiveTransform, canonical_cell_polygon, warp_image, warp_points


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
    iou_threshold: float
    image_size: int
    smoothing_window: int
    auto_max_index: int
    headless: bool
    device: str | None


@dataclass(frozen=True)
class FrameAnalysis:
    frame: np.ndarray
    warped_frame: np.ndarray
    board_result: BoardDetectionResult
    transform: PerspectiveTransform
    observation: BoardObservation
    decision: MoveDecision
    inference_ms: float
    fps: float
    detections: tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tic-tac-toe vision pipeline")
    parser.add_argument("--camera", default="auto",
                        help="Camera source: index, /dev/videoX, or auto")
    parser.add_argument(
        "--weights",
        default="runs/detect/train/weights/best.pt",
        help="YOLO weights path (.pt, .onnx, or .tflite)",
    )
    parser.add_argument("--ai-color", default="Y",
                        choices=("R", "Y"), help="Which player the AI controls")
    parser.add_argument("--board-size", type=int, default=640,
                        help="Canonical warped board size in pixels")
    parser.add_argument("--frame-width", type=int,
                        default=1280, help="Requested capture width")
    parser.add_argument("--frame-height", type=int,
                        default=720, help="Requested capture height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Requested camera frame rate")
    parser.add_argument("--confidence-threshold", type=float,
                        default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou-threshold", type=float,
                        default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--image-size", type=int, default=640,
                        help="YOLO inference image size")
    parser.add_argument("--smoothing-window", type=int,
                        default=3, help="Majority-vote history window")
    parser.add_argument("--auto-max-index", type=int, default=10,
                        help="Highest integer camera index to probe")
    parser.add_argument("--device", default=None,
                        help="Ultralytics device string, for example cpu or 0")
    parser.add_argument("--headless", action="store_true",
                        help="Disable the preview window")
    return parser


def analyze_frame(
    frame: np.ndarray,
    detector: YoloInference,
    board_detector: BoardDetector,
    board_estimator: BoardStateEstimator,
    ai_color: str,
    board_size: int,
) -> FrameAnalysis:
    board_result = board_detector.detect(frame)
    transform = board_result.build_transform(board_size)
    warped_frame = warp_image(frame, transform)

    start = time.perf_counter()
    detections = detector.predict(warped_frame)
    observation = board_estimator.estimate(warped_frame, detections)
    decision = recommend_move(observation.board, ai_player=ai_color)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

    return FrameAnalysis(
        frame=frame,
        warped_frame=warped_frame,
        board_result=board_result,
        transform=transform,
        observation=observation,
        decision=decision,
        inference_ms=elapsed_ms,
        fps=fps,
        detections=tuple(detections),
    )


def _draw_panel(frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
    rendered = frame.copy()
    height, width = rendered.shape[:2]

    panel_width = min(400, max(280, width // 3))
    panel_height = min(220, max(180, height // 4))
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
    cv2.putText(rendered, f"Inference: {analysis.inference_ms:.1f} ms", (
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

    return rendered


def run_app(config: AppConfig) -> int:
    camera_source = parse_camera_source(config.camera)
    session = open_camera(camera_source, auto_max_index=config.auto_max_index)
    configure_capture(session.capture, config.frame_width,
                      config.frame_height, config.fps)

    yolo = YoloInference(
        weights_path=config.weights,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        image_size=config.image_size,
        device=config.device,
    )
    board_detector = BoardDetector()
    board_estimator = BoardStateEstimator(
        smoothing_window=config.smoothing_window)

    window_name = "Tic-Tac-Toe Vision"
    display_enabled = not config.headless
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_signature: tuple[tuple[str, ...], ...] | None = None

    try:
        while True:
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
        iou_threshold=args.iou_threshold,
        image_size=args.image_size,
        smoothing_window=args.smoothing_window,
        auto_max_index=args.auto_max_index,
        headless=args.headless,
        device=args.device,
    )
    return run_app(config)


if __name__ == "__main__":
    raise SystemExit(main())
