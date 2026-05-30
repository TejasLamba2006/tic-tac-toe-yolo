from __future__ import annotations

from pathlib import Path

import numpy as np

from ..vision.board_state import Detection


class YoloInference:
    """Thin wrapper around Ultralytics YOLO inference.

    The wrapper keeps the rest of the pipeline agnostic to the exact model format.
    Ultralytics can load PyTorch, ONNX, and TFLite weights through the same API.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import failure is environment specific
            raise RuntimeError(
                "Ultralytics is required for the local inference path. Install it before running main.py."
            ) from exc

        self.model = YOLO(str(self.weights_path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single frame and return parsed detections."""

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
