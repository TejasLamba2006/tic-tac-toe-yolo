"""
yolo_infer.py — Standalone YOLO inference wrapper for TicTacToe
Supports: ultralytics (.pt), TFLite (.tflite), ONNX (.onnx), STM32 NPU (.nb)
Ported from the STM32M257f-dk-x-linux-ai reference project (no relative imports).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Detection dataclass (shared between vision and AI layers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """Single YOLO detection."""
    class_id: int
    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ---------------------------------------------------------------------------
# NMS helpers
# ---------------------------------------------------------------------------

def _compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


def _nms(boxes, scores, iou_threshold: float = 0.45) -> List[int]:
    order: deque = deque(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True))
    keep = []
    while order:
        current = order.popleft()
        keep.append(current)
        order = deque(idx for idx in order if _compute_iou(boxes[current], boxes[idx]) < iou_threshold)
    return keep


def _as_numpy_dtype(dtype) -> np.dtype:
    return np.dtype(dtype)


def _is_float_dtype(dtype) -> bool:
    return np.issubdtype(_as_numpy_dtype(dtype), np.floating)


def _select_yolo_output(output_tensor: np.ndarray) -> np.ndarray:
    if output_tensor.ndim == 3:
        if output_tensor.shape[1] < output_tensor.shape[2]:
            return output_tensor[0]
        return output_tensor[0].T
    return output_tensor


# ---------------------------------------------------------------------------
# YoloInference
# ---------------------------------------------------------------------------

class YoloInference:
    """
    Thin wrapper supporting:
      - ultralytics YOLO (.pt)
      - TFLite (.tflite)
      - ONNX (.onnx)
      - STM32 NPU native (.nb)

    Class IDs (tic-tac-toe model):
      0 = empty
      1 = red_ball   (Human / X)
      2 = yellow_ball (Robot / O)
    """

    DEFAULT_CLASSES = {0: "empty", 1: "red_ball", 2: "yellow_ball"}

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        image_size: int = 320,
        device: str | None = None,
        use_npu: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.use_npu = use_npu
        self.last_model_run_ms: float | None = None
        self.class_names = dict(self.DEFAULT_CLASSES)

        suffix = self.weights_path.suffix.lower()
        if suffix == ".tflite":
            self.model_type = "tflite"
            self._init_tflite()
        elif suffix == ".onnx":
            self.model_type = "onnx"
            self._init_onnx()
        elif suffix == ".nb":
            self.model_type = "nb"
            self._init_nb()
        else:
            self.model_type = "ultralytics"
            self._init_ultralytics()

        print(f"[YOLO] Loaded {self.model_type} model: {self.weights_path.name}")
        print(f"[YOLO] Classes: {self.class_names}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_tflite(self) -> None:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
        except ImportError:
            try:
                import tensorflow.lite as tflite  # type: ignore
            except ImportError as exc:
                raise RuntimeError("tflite_runtime or tensorflow required for .tflite models") from exc

        delegates = []
        if self.use_npu:
            delegate_path = "/usr/lib/libvx_delegate.so.2"
            if Path(delegate_path).exists():
                try:
                    delegates = [tflite.load_delegate(delegate_path)]
                    print(f"[YOLO] NPU delegate loaded: {delegate_path}")
                except Exception as e:
                    print(f"[YOLO] NPU delegate failed ({e}), using CPU")
            else:
                print(f"[YOLO] NPU delegate not found at {delegate_path}, using CPU")

        self.interpreter = tflite.Interpreter(
            model_path=str(self.weights_path),
            num_threads=os.cpu_count(),
            experimental_delegates=delegates,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _init_onnx(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError("onnxruntime required for .onnx models") from exc

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["VSINPUExecutionProvider"] if self.use_npu else []
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.weights_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        try:
            meta = self.session.get_modelmeta()
            if meta and meta.custom_metadata_map and "names" in meta.custom_metadata_map:
                import ast
                names = ast.literal_eval(meta.custom_metadata_map["names"])
                if isinstance(names, dict):
                    self.class_names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass

    def _init_nb(self) -> None:
        try:
            from stai_mpu import stai_mpu_network  # type: ignore
        except ImportError as exc:
            raise RuntimeError("stai_mpu required for .nb models") from exc
        self.stai_mpu_model = stai_mpu_network(
            model_path=str(self.weights_path),
            use_hw_acceleration=self.use_npu,
        )
        self.input_tensor_infos = self.stai_mpu_model.get_input_infos()
        self.input_shape = self.input_tensor_infos[0].get_shape()

    def _init_ultralytics(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ultralytics required for .pt models") from exc
        self.model = YOLO(str(self.weights_path))
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        else:
            self.class_names = {i: str(n) for i, n in enumerate(names)}

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame and return detections."""
        if self.model_type == "tflite":
            return self._predict_tflite(frame)
        elif self.model_type == "onnx":
            return self._predict_onnx(frame)
        elif self.model_type == "nb":
            return self._predict_nb(frame)
        return self._predict_ultralytics(frame)

    # ------------------------------------------------------------------
    # Backend predict implementations
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray, net_w: int, net_h: int, is_bchw: bool, input_dtype) -> np.ndarray:
        resized = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if _is_float_dtype(input_dtype):
            data = rgb.astype(np.float32) / 255.0
        else:
            data = rgb.astype(_as_numpy_dtype(input_dtype))
        if is_bchw:
            data = np.transpose(data, (2, 0, 1))
        return np.expand_dims(data, axis=0)

    def _net_dims(self, shape) -> Tuple[int, int, bool]:
        if shape[1] in (1, 3):
            return int(shape[2]), int(shape[3]), True  # NCHW
        return int(shape[1]), int(shape[2]), False      # NHWC

    def _predict_tflite(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        net_h, net_w, is_bchw = self._net_dims(self.input_details[0]["shape"])
        dtype = self.input_details[0]["dtype"]
        data = self._preprocess(frame, net_w, net_h, is_bchw, dtype)
        self.interpreter.set_tensor(self.input_details[0]["index"], data)
        t0 = time.perf_counter()
        self.interpreter.invoke()
        self.last_model_run_ms = (time.perf_counter() - t0) * 1000.0
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return self._decode(output, w, h, net_w, net_h)

    def _predict_onnx(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        net_h, net_w, is_bchw = self._net_dims(self.input_shape)
        dtype = np.float32
        data = self._preprocess(frame, net_w, net_h, is_bchw, dtype)
        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: data})
        self.last_model_run_ms = (time.perf_counter() - t0) * 1000.0
        return self._decode(outputs[0], w, h, net_w, net_h)

    def _predict_nb(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        net_h, net_w, is_bchw = self._net_dims(self.input_shape)
        dtype = self.input_tensor_infos[0].get_dtype()
        data = self._preprocess(frame, net_w, net_h, is_bchw, dtype)
        self.stai_mpu_model.set_input(0, data)
        t0 = time.perf_counter()
        self.stai_mpu_model.run()
        self.last_model_run_ms = (time.perf_counter() - t0) * 1000.0
        output = self.stai_mpu_model.get_output(index=0)
        return self._decode(output, w, h, net_w, net_h)

    def _predict_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        self.last_model_run_ms = (time.perf_counter() - t0) * 1000.0
        if not results:
            return []
        return self._parse_ultralytics(results[0])

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _decode(self, output_tensor: np.ndarray, fw: int, fh: int, nw: int, nh: int) -> List[Detection]:
        output = _select_yolo_output(output_tensor)
        boxes = output[:4, :].T      # [N, 4] cx cy w h
        scores = output[4:, :].T     # [N, num_classes]

        sx = fw / nw
        sy = fh / nh
        if boxes.size > 0 and float(np.nanmax(np.abs(boxes))) <= 2.0:
            sx, sy = float(fw), float(fh)

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        mask = confidences >= self.confidence_threshold
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

        nms_boxes = []
        for box in boxes:
            cx, cy, bw, bh = box
            nms_boxes.append([
                float((cx - bw / 2) * sx),
                float((cy - bh / 2) * sy),
                float((cx + bw / 2) * sx),
                float((cy + bh / 2) * sy),
            ])

        indices = _nms(nms_boxes, confidences.tolist(), self.iou_threshold)
        detections = []
        for idx in indices:
            x1, y1, x2, y2 = nms_boxes[idx]
            cid = int(class_ids[idx])
            conf = float(confidences[idx])
            label = self.class_names.get(cid, str(cid))
            detections.append(Detection(class_id=cid, label=label, confidence=conf, xyxy=(x1, y1, x2, y2)))
        return detections

    def _parse_ultralytics(self, result) -> List[Detection]:
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            label = self.class_names.get(cid, str(cid))
            detections.append(Detection(class_id=cid, label=label, confidence=conf, xyxy=(x1, y1, x2, y2)))
        return detections
