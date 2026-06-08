from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _load_tflite():
    try:
        import tflite_runtime.interpreter as tflite

        return tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite

            return tflite
        except ImportError as exc:
            raise RuntimeError("Install tflite_runtime or tensorflow to inspect TFLite models") from exc


def _prepare_input(image_path: Path, input_detail: dict) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    shape = input_detail["shape"]
    if shape[1] in (1, 3):
        height, width = int(shape[2]), int(shape[3])
        nchw = True
    else:
        height, width = int(shape[1]), int(shape[2])
        nchw = False

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    dtype = np.dtype(input_detail["dtype"])
    if np.issubdtype(dtype, np.floating):
        tensor = rgb.astype(np.float32) / 255.0
    else:
        tensor = rgb.astype(dtype)

    if nchw:
        tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0)


def _select_yolo_output(output: np.ndarray) -> np.ndarray:
    if output.ndim == 3:
        if output.shape[1] < output.shape[2]:
            return output[0]
        return output[0].T
    return output


def diagnose(model_path: Path, image_path: Path, threshold: float) -> int:
    tflite = _load_tflite()
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    input_tensor = _prepare_input(image_path, input_detail)
    interpreter.set_tensor(input_detail["index"], input_tensor)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_detail["index"])
    output = _select_yolo_output(output_tensor)

    print("Input:")
    print(f"  shape={input_tensor.shape} dtype={input_tensor.dtype}")
    print(f"  min={float(input_tensor.min()):.6f} max={float(input_tensor.max()):.6f}")
    print(f"  detail={input_detail}")
    print("Output:")
    print(f"  shape={output_tensor.shape} dtype={output_tensor.dtype}")
    print(f"  min={float(output_tensor.min()):.6f} max={float(output_tensor.max()):.6f}")
    print(f"  detail={output_detail}")
    print(f"  first20={output_tensor.flatten()[:20]}")

    for channel_index in range(output.shape[0]):
        channel = output[channel_index, :]
        print(
            f"  ch{channel_index}: min={float(channel.min()):.6f} "
            f"max={float(channel.max()):.6f} mean={float(channel.mean()):.6f}"
        )

    if output.shape[0] <= 4:
        raise RuntimeError(f"Expected YOLO output channels > 4, got {output.shape}")

    scores = output[4:, :].T
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)
    keep = np.where(max_scores >= threshold)[0]
    print(f"Decoded candidates >= {threshold}: {len(keep)}")
    for index in keep[:20]:
        print(
            f"  idx={int(index)} class={int(class_ids[index])} "
            f"score={float(max_scores[index]):.6f} box={output[:4, index]}"
        )

    class_max = float(scores.max())
    if class_max < threshold:
        print(
            f"FAIL: highest class score {class_max:.6f} is below threshold {threshold}. "
            "Do not generate .nb from this TFLite."
        )
        return 2

    print("PASS: class scores survived quantization.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect YOLO TFLite tensor ranges and decoded scores")
    parser.add_argument("model", help="Path to .tflite model")
    parser.add_argument("image", help="Image used for a quick inference sanity check")
    parser.add_argument("--threshold", type=float, default=0.25, help="Minimum detection score")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return diagnose(Path(args.model), Path(args.image), args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
