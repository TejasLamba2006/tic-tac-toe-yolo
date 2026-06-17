from __future__ import annotations

import os
from collections import deque
from pathlib import Path
import time

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
    order: deque[int] = deque(
        sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )
    )

    keep = []

    while order:
        current = order.popleft()
        keep.append(current)

        order = deque(
            idx
            for idx in order
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold
        )

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


def _debug_tensor(prefix: str, tensor: np.ndarray) -> None:
    if not os.environ.get("YOLO_DEBUG"):
        return
    # print(
    #     f"[YOLO_DEBUG] {prefix}: shape={tensor.shape} dtype={tensor.dtype} "
    #     f"min={float(np.min(tensor)):.6f} max={float(np.max(tensor)):.6f}"
    # )
    # print(f"[YOLO_DEBUG] {prefix} first20={tensor.flatten()[:20]}")
    if tensor.ndim == 3:
        for channel_index in range(tensor.shape[1]):
            channel = tensor[0, channel_index, :]
            # print(
            #     f"[YOLO_DEBUG] {prefix} ch{channel_index}: "
            #     f"min={float(channel.min()):.6f} "
            #     f"max={float(channel.max()):.6f} "
            #     f"mean={float(channel.mean()):.6f}"
            # )


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
        self.last_model_run_ms: float | None = None

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

    def _init_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for ONNX inference. Install it before running."
            ) from exc

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if self.use_npu:
            providers.append('VSINPUExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(
            str(self.weights_path),
            sess_options=session_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Tic-tac-toe classes
        self.class_names = {0: "empty", 1: "red_ball", 2: "yellow_ball"}
        try:
            meta = self.session.get_modelmeta()
            if meta and meta.custom_metadata_map and 'names' in meta.custom_metadata_map:
                import ast
                names = ast.literal_eval(meta.custom_metadata_map['names'])
                if isinstance(names, dict):
                    self.class_names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass

    def _init_nb(self) -> None:
        try:
            from stai_mpu import stai_mpu_network
        except ImportError as exc:
            raise RuntimeError(
                "stai_mpu library is required for running native .nb models on the board."
            ) from exc

        # Initialize the network binary model
        self.stai_mpu_model = stai_mpu_network(
            model_path=str(self.weights_path),
            use_hw_acceleration=self.use_npu
        )
        self.input_tensor_infos = self.stai_mpu_model.get_input_infos()
        self.input_shape = self.input_tensor_infos[0].get_shape()

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
        if self.model_type == "tflite":
            return self._predict_tflite(frame)
        elif self.model_type == "onnx":
            return self._predict_onnx(frame)
        elif self.model_type == "nb":
            return self._predict_nb(frame)
        return self._predict_ultralytics(frame)

    def _predict_onnx(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]

        # Determine if the layout is NCHW or NHWC
        if self.input_shape[1] == 3 or self.input_shape[1] == 1:
            # NCHW
            net_h, net_w = self.input_shape[2], self.input_shape[3]
            is_bchw = True
        else:
            # NHWC
            net_h, net_w = self.input_shape[1], self.input_shape[2]
            is_bchw = False

        # Preprocess frame (resize, RGB, normalization)
        resized = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.float32) / 255.0

        if is_bchw:
            input_data = np.transpose(input_data, (2, 0, 1))

        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_data})
        self.last_model_run_ms = (time.perf_counter() - start) * 1000.0
        output_tensor = outputs[0]

        _debug_tensor("onnx_input", input_data)
        _debug_tensor("onnx_output", output_tensor)
        return self._decode_yolo_output(output_tensor, w, h, net_w, net_h)

    def _predict_nb(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]

        input_shape = self.input_shape
        if input_shape[1] == 3 or input_shape[1] == 1:
            # NCHW
            net_h, net_w = input_shape[2], input_shape[3]
            is_bchw = True
        else:
            # NHWC
            net_h, net_w = input_shape[1], input_shape[2]
            is_bchw = False

        resized = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_dtype = self.input_tensor_infos[0].get_dtype()
        if _is_float_dtype(input_dtype):
            input_data = rgb.astype(np.float32) / 255.0
        else:
            input_data = rgb.astype(_as_numpy_dtype(input_dtype))

        if is_bchw:
            input_data = np.transpose(input_data, (2, 0, 1))

        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        self.stai_mpu_model.set_input(0, input_data)
        start = time.perf_counter()
        self.stai_mpu_model.run()
        self.last_model_run_ms = (time.perf_counter() - start) * 1000.0

        # Retrieve output
        output_tensor = self.stai_mpu_model.get_output(index=0)

        _debug_tensor("nb_input", input_data)
        _debug_tensor("nb_output", output_tensor)
        return self._decode_yolo_output(output_tensor, w, h, net_w, net_h)

    def _predict_ultralytics(self, frame: np.ndarray) -> list[Detection]:
        start = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        self.last_model_run_ms = (time.perf_counter() - start) * 1000.0
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
        if _is_float_dtype(input_dtype):
            input_data = rgb.astype(np.float32) / 255.0
        else:
            input_data = rgb.astype(_as_numpy_dtype(input_dtype))

        if is_bchw:
            input_data = np.transpose(input_data, (2, 0, 1))

        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        start = time.perf_counter()
        self.interpreter.invoke()
        self.last_model_run_ms = (time.perf_counter() - start) * 1000.0

        # Retrieve and transpose output
        output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        _debug_tensor("tflite_input", input_data)
        _debug_tensor("tflite_output", output_tensor)
        return self._decode_yolo_output(output_tensor, w, h, net_w, net_h)

    def _decode_yolo_output(
        self,
        output_tensor: np.ndarray,
        frame_width: int,
        frame_height: int,
        net_width: int,
        net_height: int,
    ) -> list[Detection]:
        output = _select_yolo_output(output_tensor)
        boxes = output[:4, :].T  # [num_boxes, 4] (x_center, y_center, width, height)
        scores = output[4:, :].T  # [num_boxes, num_classes]
        coordinate_scale_x = frame_width / net_width
        coordinate_scale_y = frame_height / net_height
        if boxes.size > 0 and float(np.nanmax(np.abs(boxes))) <= 2.0:
            coordinate_scale_x = frame_width
            coordinate_scale_y = frame_height

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        if os.environ.get("YOLO_DEBUG"):
            print(
                f"[YOLO_DEBUG] coordinate_scale_x={coordinate_scale_x:.6f} "
                f"coordinate_scale_y={coordinate_scale_y:.6f}"
            )
            print("[YOLO_DEBUG] decoded candidates before threshold:")
            for index in np.argsort(confidences)[::-1][:20]:
                print(
                    f"[YOLO_DEBUG] idx={int(index)} class={int(class_ids[index])} "
                    f"conf={float(confidences[index]):.6f} box={boxes[index]}"
                )

        mask = confidences >= self.confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        nms_boxes = []
        for box in boxes:
            x_c, y_c, box_w, box_h = box

            x1 = (x_c - box_w / 2) * coordinate_scale_x
            y1 = (y_c - box_h / 2) * coordinate_scale_y
            x2 = (x_c + box_w / 2) * coordinate_scale_x
            y2 = (y_c + box_h / 2) * coordinate_scale_y
            nms_boxes.append([float(x1), float(y1), float(x2), float(y2)])

        indices = nms(
            nms_boxes,
            confidences.tolist(),
            self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            for idx in indices:
                x1, y1, x2, y2 = nms_boxes[idx]
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
        if os.environ.get("YOLO_DEBUG"):
            print(f"[YOLO_DEBUG] detections after nms={len(detections)}")
            for detection in detections[:20]:
                print(
                    f"[YOLO_DEBUG] det label={detection.label} "
                    f"conf={detection.confidence:.6f} xyxy={detection.xyxy}"
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
