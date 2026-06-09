#!/usr/bin/env python3
"""
detect.py — Minimal YOLO live detection viewer.

Opens a camera, runs the YOLO model every frame, and draws colour-coded
bounding boxes + confidence scores in an OpenCV window.

No game logic, no GTK, no robot arm — detection only.

Usage (on device with NPU):
    python3 detect.py \
        --weights tictactoe_yolov8_quant_pc_uf_od_tictactoe_1.nb \
        --camera auto \
        --npu

Usage (on PC with .pt weights):
    python3 detect.py --weights yolov8n.pt --camera 0
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.ai.yolo_inference import YoloInference
from src.vision.camera import configure_capture, open_camera, parse_camera_source

# ---------------------------------------------------------------------------
# Label → BGR colour map
# ---------------------------------------------------------------------------
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "red_ball":    (0,   60,  220),   # red   (BGR)
    "yellow_ball": (0,  210,  255),   # yellow (BGR)
    "empty":       (120, 120, 120),   # grey
}

def _color(label: str, class_id: int) -> tuple[int, int, int]:
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    seed = (class_id * 73) % 255
    return (seed, 255 - seed, 160)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_boxes(frame: np.ndarray, detections, conf_threshold: float) -> np.ndarray:
    """Draw a bounding box + label for every detection above the threshold."""
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        if det.confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = det.xyxy
        l = int(np.clip(x1, 0, w - 1))
        t = int(np.clip(y1, 0, h - 1))
        r = int(np.clip(x2, 0, w - 1))
        b = int(np.clip(y2, 0, h - 1))

        color = _color(det.label, det.class_id)

        # Box
        cv2.rectangle(out, (l, t), (r, b), color, 2, cv2.LINE_AA)

        # Label background
        label_text = f"{det.label}  {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(t - 4, th + 4)
        cv2.rectangle(out, (l, label_y - th - 4), (l + tw + 6, label_y + 2), color, -1)

        # Label text
        cv2.putText(out, label_text, (l + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_hud(frame: np.ndarray, fps: float, infer_ms: float,
             n_dets: int, npu: bool) -> np.ndarray:
    """Overlay a small stats panel in the top-left corner."""
    out = frame.copy()
    lines = [
        f"FPS:      {fps:5.1f}",
        f"Infer:    {infer_ms:5.1f} ms",
        f"Detects:  {n_dets}",
        f"Backend:  {'NPU' if npu else 'CPU'}",
    ]
    pad = 10
    line_h = 22
    panel_h = line_h * len(lines) + pad
    panel_w = 190
    cv2.rectangle(out, (0, 0), (panel_w, panel_h), (18, 18, 18), -1)
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (pad, pad + (i + 1) * line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ---- Load model --------------------------------------------------------
    print(f"[DETECT] Loading weights: {args.weights}  (NPU={args.npu})")
    model = YoloInference(
        weights_path=args.weights,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.image_size,
        use_npu=args.npu,
    )
    print("[DETECT] Model loaded.")

    # ---- Open camera -------------------------------------------------------
    source = parse_camera_source(args.camera)
    cap = open_camera(source)
    if cap is None or not cap.isOpened():
        print(f"[DETECT] ERROR: Could not open camera '{args.camera}'")
        return
    configure_capture(cap, args.width, args.height, args.fps)
    print(f"[DETECT] Camera ready — {args.width}×{args.height} @ {args.fps}fps")

    # ---- Window ------------------------------------------------------------
    win = "YOLO Detection — press Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps_start = time.time()
    fps_count = 0
    fps_value = 0.0
    infer_ms  = 0.0

    print("[DETECT] Running. Press Q or ESC to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            cv2.waitKey(10)
            continue

        # ---- Inference -----------------------------------------------------
        t0 = time.perf_counter()
        detections = model.predict(frame)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # ---- FPS -----------------------------------------------------------
        fps_count += 1
        if fps_count >= 20:
            elapsed = time.time() - fps_start
            fps_value = fps_count / elapsed if elapsed > 0 else 0.0
            fps_start = time.time()
            fps_count = 0

        # ---- Draw ----------------------------------------------------------
        display = draw_boxes(frame, detections, args.conf)
        display = draw_hud(display, fps_value, infer_ms, len(detections), args.npu)

        cv2.imshow(win, display)

        # ---- Key handling --------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DETECT] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone YOLO live detection with OpenCV bounding boxes")
    p.add_argument("--weights",     default="yolov8n.pt",
                   help="Model weights: .pt / .onnx / .tflite / .nb")
    p.add_argument("--camera",      default="auto",
                   help="Camera source: index, /dev/videoX, or 'auto'")
    p.add_argument("--width",       type=int,   default=640)
    p.add_argument("--height",      type=int,   default=480)
    p.add_argument("--fps",         type=int,   default=30)
    p.add_argument("--conf",        type=float, default=0.50,
                   help="Confidence threshold for displayed boxes")
    p.add_argument("--iou",         type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--image-size",  type=int,   default=320,
                   help="YOLO input resolution")
    p.add_argument("--npu",         action="store_true",
                   help="Enable STM32 NPU hardware acceleration")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
