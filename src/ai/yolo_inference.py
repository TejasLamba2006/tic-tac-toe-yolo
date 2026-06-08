from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ..vision.board_state import Detection


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def nms(boxes, scores, iou_threshold=0.45):
    order = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    keep = []

    while order:
        current = order.pop(0)
        keep.append(current)

        order = [
            idx
            for idx in order
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold
        ]

    return keep


class YoloInference:
    """Thin wrapper around Ultralytics YOLO and TFLite inference.

    The wrapper keeps the rest of the pipeline agnostic to the exact model format.
    On the board, it loads .tflite weights without importing ultralytics.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
        use_npu: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.use_npu = use_npu

        if self.weights_path.suffix.lower() == ".tflite":
            self.use_tflite = True
            self._init_tflite()
        else:
            self.use_tflite = False
            self._init_ultralytics()

    def _init_tflite(self) -> None:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError as exc:
                raise RuntimeError(
                    "Either tflite_runtime or tensorflow is required to load TFLite models on the board"
                ) from exc

        delegates = []
        if self.use_npu:
            delegate_path = '/usr/lib/libvx_delegate.so.2'
            if Path(delegate_path).exists():
                try:
                    delegates = [tflite.load_delegate(delegate_path)]
                    print(f"Successfully loaded NPU delegate from {delegate_path}")
                except Exception as e:
                    print(f"Warning: Failed to load NPU delegate: {e}. Falling back to CPU.")
            else:
                print(f"Warning: NPU delegate not found at {delegate_path}. Falling back to CPU.")

        self.interpreter = tflite.Interpreter(
            model_path=str(self.weights_path),
            num_threads=os.cpu_count(),
            experimental_delegates=delegates
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Tic-tac-toe classes
        self.class_names = {0: "empty", 1: "red_ball", 2: "yellow_ball"}

    def _init_ultralytics(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import failure is environment specific
            raise RuntimeError(
                "Ultralytics is required for the local inference path. Install it before running main.py."
            ) from exc

        self.model = YOLO(str(self.weights_path))
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single frame and return parsed detections."""
        if self.use_tflite:
            return self._predict_tflite(frame)
        return self._predict_ultralytics(frame)

    def _predict_ultralytics(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._parse_result(results[0])

    def _predict_tflite(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        input_shape = self.input_details[0]['shape']

        # Determine if the layout is NCHW or NHWC
        if input_shape[1] == 3 or input_shape[1] == 1:
            # NCHW
            net_h, net_w = input_shape[2], input_shape[3]
            is_bchw = True
        else:
            # NHWC
            net_h, net_w = input_shape[1], input_shape[2]
            is_bchw = False

        # Preprocess frame (resize, RGB, normalization)
        resized = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.float32:
            input_data = rgb.astype(np.float32) / 255.0
        else:
            input_data = rgb.astype(input_dtype)

        if is_bchw:
            input_data = np.transpose(input_data, (2, 0, 1))

        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Retrieve and transpose output
        output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_shape = output_tensor.shape
        if len(output_shape) == 3:
            if output_shape[1] < output_shape[2]:
                # Shape is [1, 7, 8400] -> transpose to [7, 8400]
                output = output_tensor[0]
            else:
                # Shape is [1, 8400, 7] -> transpose to [7, 8400]
                output = output_tensor[0].T
        else:
            # Handle squeeze/unsqueeze if output has different dimensions
            output = output_tensor

        boxes = output[:4, :].T  # [num_boxes, 4] (x_center, y_center, width, height)
        scores = output[4:, :].T  # [num_boxes, num_classes]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        nms_boxes = []
        for box in boxes:
            x_c, y_c, box_w, box_h = box
            scale_x = w / net_w
            scale_y = h / net_h

            x1 = (x_c - box_w / 2) * scale_x
            y1 = (y_c - box_h / 2) * scale_y
            box_width = box_w * scale_x
            box_height = box_h * scale_y
            nms_boxes.append([float(x1), float(y1), float(box_width), float(box_height)])

        indices = nms(
            nms_boxes,
            confidences.tolist(),
            self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, box_width, box_height = nms_boxes[idx]
                x2 = x1 + box_width
                y2 = y1 + box_height
                class_id = int(class_ids[idx])
                conf = float(confidences[idx])
                label = self.class_names.get(class_id, str(class_id))
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=label,
                        confidence=conf,
                        xyxy=(x1, y1, x2, y2),
                    )
                )
        return detections

    def _parse_result(self, result) -> list[Detection]:
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            label = self.class_names.get(class_id, str(class_id))
            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=confidence,
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections
