from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO


def export_quantization_friendly_saved_model(
    weights: Path,
    data: Path,
    image_size: int,
) -> Path:
    """Export with Ultralytics int8=True so the SavedModel is ST-quantization friendly."""

    model = YOLO(str(weights))
    exported = model.export(
        format="tflite",
        imgsz=image_size,
        int8=True,
        data=str(data),
    )

    exported_path = Path(str(exported))
    if exported_path.is_dir():
        return exported_path
    return exported_path.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Ultralytics SavedModel that ST Model Zoo Services expects "
            "as input for YOLOv8 quantization."
        )
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/train-5/weights/best.pt",
        help="Path to the trained YOLOv8 .pt weights",
    )
    parser.add_argument(
        "--data",
        default="data.yaml",
        help="Ultralytics dataset yaml used for calibration metadata",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Input size used during training and deployment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    saved_model_dir = export_quantization_friendly_saved_model(
        weights=Path(args.weights),
        data=Path(args.data),
        image_size=args.image_size,
    )
    print(f"SavedModel directory: {saved_model_dir}")
    print("Next ST quantization step:")
    print("  python scripts/tflite_quant.py --config-name config_quant.yaml --config-path .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
