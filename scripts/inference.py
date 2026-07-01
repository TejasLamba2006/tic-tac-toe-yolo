#!/usr/bin/env python3
"""Interactive inference script — pick a model, pick a camera, run detection.

Scans ``artifacts/`` for trained weights, lists available cameras via OpenCV,
then runs live YOLO detection with an OpenCV preview window.

Usage:
    python scripts/inference.py
    python scripts/inference.py --model artifacts/training/weights/best.pt
    python scripts/inference.py --camera 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# ── Camera probing ──────────────────────────────────────────────────────────

def _probe_cameras(max_scan: int = 10) -> list[int]:
    """Return indices of cameras that can be opened."""
    available: list[int] = []
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                available.append(idx)
    return available


def _choose_camera(prompt: bool = True) -> int:
    """Interactively ask the user to pick a camera."""
    cameras = _probe_cameras()
    if not cameras:
        print("No cameras detected.")
        return -1

    print("\nAvailable cameras:")
    for i, idx in enumerate(cameras):
        cap = cv2.VideoCapture(idx)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"  [{i}]  Camera {idx}   ({w}x{h})")

    if len(cameras) == 1:
        print(f"\nAuto-selected camera {cameras[0]} (only option).")
        return cameras[0]

    while True:
        try:
            choice = int(input(f"\nPick camera [0-{len(cameras) - 1}]: "))
            if 0 <= choice < len(cameras):
                return cameras[choice]
        except (ValueError, EOFError):
            pass
        print("Invalid choice, try again.")


# ── Model probing ───────────────────────────────────────────────────────────

_MODEL_EXTENSIONS = {".pt", ".onnx", ".tflite", ".nb"}


def _scan_models(artifacts_dir: Path = Path("artifacts")) -> list[Path]:
    """Recursively find model weight files under artifacts/."""
    models: list[Path] = []
    if artifacts_dir.is_dir():
        for p in sorted(artifacts_dir.rglob("*")):
            if p.suffix.lower() in _MODEL_EXTENSIONS and p.is_file():
                models.append(p)
    return models


def _choose_model(force_path: str | None = None) -> Path | None:
    """Interactively ask the user to pick a model file."""
    if force_path:
        p = Path(force_path)
        if p.is_file():
            return p
        print(f"Model not found: {force_path}")
        return None

    models = _scan_models()
    if not models:
        print("No model files found under artifacts/.  Train or export first.")
        return None

    print("\nAvailable models:")
    for i, m in enumerate(models):
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  [{i}]  {m.name:<55s}  {size_mb:7.1f} MB")

    while True:
        try:
            choice = int(input(f"\nPick model [0-{len(models) - 1}]: "))
            if 0 <= choice < len(models):
                return models[choice]
        except (ValueError, EOFError):
            pass
        print("Invalid choice, try again.")


# ── YOLO inference wrapper (lightweight, no src/ imports) ───────────────────

def _load_yolo(model_path: Path, conf: float, iou: float, imgsz: int, use_npu: bool):
    """Load a YOLO model via ultralytics and return (model, labels, is_ultralytics).

    Falls back to onnxruntime / stai_mpu for non-.pt files.
    """
    suffix = model_path.suffix.lower()

    # Ultralytics handles .pt natively and can also do .onnx inference.
    if suffix in (".pt", ".onnx"):
        try:
            from ultralytics import YOLO
            m = YOLO(str(model_path))
            labels = {i: name for i, name in m.names.items()} if hasattr(m, "names") else {}
            return m, labels, True
        except Exception:
            if suffix == ".onnx":
                pass  # fall through to onnxruntime
            else:
                raise

    # ONNX via onnxruntime
    if suffix == ".onnx":
        try:
            import onnxruntime as ort
        except ImportError:
            sys.exit("onnxruntime not installed: pip install onnxruntime")

        providers = ["CPUExecutionProvider"]
        if use_npu:
            # STM32 NPU via VSINPU EP (available on the board)
            for ep in ["VSINPUExecutionProvider", "CUDAExecutionProvider"]:
                if ep in ort.get_available_providers():
                    providers.insert(0, ep)

        sess = ort.InferenceSession(str(model_path), providers=providers)
        labels = {0: "board", 1: "red_ball", 2: "yellow_ball"}
        return sess, labels, False

    # TFLite
    if suffix == ".tflite":
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter as tflite  # type: ignore
            except ImportError:
                sys.exit("No TFLite runtime: pip install tflite-runtime")

        delegates = []
        if use_npu:
            try:
                delegates.append(tflite.load_delegate("/usr/lib/libvx_delegate.so.2"))
            except Exception:
                pass
        sess = tflite.Interpreter(model_path=str(model_path), experimental_delegates=delegates or None)
        sess.allocate_tensors()
        labels = {0: "board", 1: "red_ball", 2: "yellow_ball"}
        return sess, labels, False

    # Neural Binary (.nb)
    if suffix == ".nb":
        try:
            from stai_mpu import stai_mpu_network  # type: ignore
        except ImportError:
            sys.exit("stai_mpu not installed (STM32 only)")

        sess = stai_mpu_network(model_path=str(model_path), use_hw_acceleration=use_npu)
        labels = {0: "board", 1: "red_ball", 2: "yellow_ball"}
        return sess, labels, False

    sys.exit(f"Unsupported model format: {suffix}")


def _ultralytics_predict(model, frame, conf: float, iou: float, imgsz: int):
    """Run inference via ultralytics and return list of (x1,y1,x2,y2, cls, conf)."""
    results = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    detections: list[tuple] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            c = float(box.conf[0].cpu().numpy())
            detections.append((int(x1), int(y1), int(x2), int(y2), cls, c))
    return detections


def _onnx_predict(sess, frame, conf: float, imgsz: int):
    """Run inference via onnxruntime and return list of (x1,y1,x2,y2, cls, conf)."""
    import numpy as np

    img_in = cv2.resize(frame, (imgsz, imgsz))
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_in = img_in.transpose(2, 0, 1)[None]  # NCHW

    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: img_in})[0]  # shape (1, 84, 8400) or similar

    detections: list[tuple] = []
    h, w = frame.shape[:2]
    sx, sy = w / imgsz, h / imgsz

    # YOLO v8 output: (1, 4+nc, N) — first 4 are cx,cy,w,h, rest are class scores
    out = output[0]  # (84, 8400)
    boxes_raw = out[:4].T  # (N, 4)
    scores = out[4:].T     # (N, nc)

    for i in range(len(boxes_raw)):
        best_cls = int(scores[i].argmax())
        best_conf = float(scores[i][best_cls])
        if best_conf < conf:
            continue
        cx, cy, bw, bh = boxes_raw[i]
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        detections.append((x1, y1, x2, y2, best_cls, best_conf))
    return detections


def _tflite_predict(sess, frame, conf: float, imgsz: int):
    """Run inference via tflite_runtime and return list of (x1,y1,x2,y2, cls, conf)."""
    import numpy as np

    input_details = sess.get_input_details()
    output_details = sess.get_output_details()

    img_in = cv2.resize(frame, (imgsz, imgsz))
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, 0)

    sess.set_tensor(input_details[0]["index"], img_in)
    sess.invoke()
    out = sess.get_tensor(output_details[0]["index"])[0]

    detections: list[tuple] = []
    h, w = frame.shape[:2]
    sx, sy = w / imgsz, h / imgsz

    boxes_raw = out[:4].T
    scores = out[4:].T

    for i in range(len(boxes_raw)):
        best_cls = int(scores[i].argmax())
        best_conf = float(scores[i][best_cls])
        if best_conf < conf:
            continue
        cx, cy, bw, bh = boxes_raw[i]
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        detections.append((x1, y1, x2, y2, best_cls, best_conf))
    return detections


def _nb_predict(sess, frame, conf: float, imgsz: int):
    """Run inference via stai_mpu and return list of (x1,y1,x2,y2, cls, conf)."""
    import numpy as np

    img_in = cv2.resize(frame, (imgsz, imgsz))
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in.transpose(2, 0, 1), 0)

    outputs = sess.run([img_in])
    out = outputs[0] if isinstance(outputs, list) else outputs
    if out.ndim == 3:
        out = out[0]

    detections: list[tuple] = []
    h, w = frame.shape[:2]
    sx, sy = w / imgsz, h / imgsz

    boxes_raw = out[:4].T
    scores = out[4:].T

    for i in range(len(boxes_raw)):
        best_cls = int(scores[i].argmax())
        best_conf = float(scores[i][best_cls])
        if best_conf < conf:
            continue
        cx, cy, bw, bh = boxes_raw[i]
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        detections.append((x1, y1, x2, y2, best_cls, best_conf))
    return detections


# ── Drawing helpers ──────────────────────────────────────────────────────────

_COLORS = [
    (0, 255, 0),    # board / class 0
    (0, 0, 255),    # red_ball / class 1
    (0, 255, 255),  # yellow_ball / class 2
    (255, 128, 0),  # extra
    (128, 0, 255),  # extra
]


def _draw_detections(
    frame,
    detections: list[tuple],
    labels: dict[int, str],
    fps: float,
) -> None:
    """Draw bounding boxes, labels, and FPS on frame."""
    for x1, y1, x2, y2, cls, c in detections:
        color = _COLORS[cls % len(_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = labels.get(cls, str(cls))
        text = f"{name} {c:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    # FPS overlay
    cv2.putText(
        frame, f"FPS: {fps:.0f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA,
    )


# ── Main loop ───────────────────────────────────────────────────────────────

_COLORS_LIST = [
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def run(model_path: Path, camera_idx: int, conf: float, iou: float, imgsz: int, use_npu: bool) -> None:
    print(f"\nLoading model: {model_path.name}")
    model, labels, is_ultra = _load_yolo(model_path, conf, iou, imgsz, use_npu)
    print("Model loaded.\n")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {camera_idx}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {camera_idx}: {w}x{h}")
    print("Press Q or ESC to quit.\n")

    win = "Inference — Q / ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps_smooth = 0.0
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue

        # Dispatch to the correct predictor
        if is_ultra:
            dets = _ultralytics_predict(model, frame, conf, iou, imgsz)
        elif isinstance(model, type(None)):
            break
        else:
            suffix = model_path.suffix.lower()
            if suffix == ".onnx":
                dets = _onnx_predict(model, frame, conf, imgsz)
            elif suffix == ".tflite":
                dets = _tflite_predict(model, frame, conf, imgsz)
            elif suffix == ".nb":
                dets = _nb_predict(model, frame, conf, imgsz)
            else:
                dets = []

        dt = time.perf_counter() - t0
        fps_smooth = fps_smooth * 0.9 + (1.0 / max(dt, 1e-6)) * 0.1

        _draw_detections(frame, dets, labels, fps_smooth)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive YOLO inference with camera/model selection.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model weights (.pt/.onnx/.tflite/.nb).  "
             "Omit to pick interactively from artifacts/.",
    )
    parser.add_argument(
        "--camera", default=None,
        help="Camera index (e.g. 0).  Omit to pick interactively.",
    )
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference resolution")
    parser.add_argument("--npu", action="store_true", help="Enable STM32 NPU acceleration")
    args = parser.parse_args()

    model_path = _choose_model(args.model)
    if model_path is None:
        sys.exit(1)

    if args.camera is not None:
        cam = int(args.camera)
    else:
        cam = _choose_camera()

    if cam < 0:
        sys.exit(1)

    run(model_path, cam, args.conf, args.iou, args.imgsz, args.npu)


if __name__ == "__main__":
    main()
