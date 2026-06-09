#!/usr/bin/env python3
"""
detect.py — Display the warped board inset (grid + YOLO boxes) in an OpenCV window.

Calls analyze_frame() each tick to get a FrameAnalysis, then passes it to
_build_board_inset() from src.main and shows the result.

Usage (STM32 device, NPU):
    python3 detect.py \
        --weights tictactoe_yolov8_quant_pc_uf_od_tictactoe_1.nb \
        --camera auto \
        --npu

Usage (PC, PyTorch):
    python3 detect.py --weights yolov8n.pt --camera 0
"""
from __future__ import annotations

import argparse

import cv2

from src.ai.yolo_inference import YoloInference
from src.vision.board_detector import BoardDetector
from src.vision.board_state import BoardStateEstimator
from src.vision.camera import (
    configure_capture,
    open_camera,
    parse_camera_source,
)
from src.vision.stability import BoardGeometryTracker
from src.main import analyze_frame, _build_board_inset


def run(args: argparse.Namespace) -> None:
    # ---- Load model --------------------------------------------------------
    print(f"[DETECT] Loading: {args.weights}  (NPU={args.npu})")
    model = YoloInference(
        weights_path=args.weights,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.image_size,
        use_npu=args.npu,
    )
    print("[DETECT] Model loaded.")

    # ---- Pipeline helpers --------------------------------------------------
    board_detector   = BoardDetector()
    board_estimator  = BoardStateEstimator(
        minimum_confidence=args.conf,
        smoothing_window=5,
    )
    geometry_tracker = BoardGeometryTracker()

    # ---- Camera (returns CameraSession, not VideoCapture) ------------------
    source  = parse_camera_source(args.camera)
    session = open_camera(source)          # CameraSession — has .read() / .release()

    # configure_capture takes the inner cv2.VideoCapture + source
    configure_capture(session.capture, args.width, args.height, args.fps,
                      source=session.source)
    print(f"[DETECT] Camera ready — source={session.source}")

    # ---- Window ------------------------------------------------------------
    win = "Board inset — Q / ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[DETECT] Running. Press Q or ESC to quit.")

    while True:
        ok, frame = session.read()
        if not ok or frame is None:
            cv2.waitKey(10)
            continue

        # Full pipeline: board warp + YOLO on warped frame
        analysis = analyze_frame(
            frame=frame,
            detector=model,
            board_detector=board_detector,
            board_estimator=board_estimator,
            ai_color="Y",
            board_size=320,
            geometry_tracker=geometry_tracker,
        )

        # Warped board with 3×3 grid lines and detection bounding boxes
        inset = _build_board_inset(analysis, inset_size=args.inset_size)
        cv2.imshow(win, inset)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    session.release()
    cv2.destroyAllWindows()
    print("[DETECT] Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Show _build_board_inset output in an OpenCV window")
    p.add_argument("--weights",    default="yolov8n.pt",
                   help="Model weights: .pt / .onnx / .tflite / .nb")
    p.add_argument("--camera",     default="auto",
                   help="Camera source: index, /dev/videoX, or 'auto'")
    p.add_argument("--width",      type=int, default=640)
    p.add_argument("--height",     type=int, default=480)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--conf",       type=float, default=0.50,
                   help="YOLO confidence threshold")
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--image-size", type=int, default=320,
                   help="YOLO input resolution")
    p.add_argument("--inset-size", type=int, default=480,
                   help="Size of the displayed board inset window (pixels)")
    p.add_argument("--npu",        action="store_true",
                   help="Enable STM32 NPU acceleration")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
