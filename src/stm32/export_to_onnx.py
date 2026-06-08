from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence


def export_onnx_model(weights_path: str | Path, output_path: str | Path | None = None, imgsz: int = 640, opset: int = 13) -> Path:
    """Export an Ultralytics model to ONNX and return the final path."""

    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Ultralytics is required to export ONNX models") from exc

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=False
    )
    exported_path = Path(str(exported))

    if output_path is None:
        return exported_path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != destination.resolve():
        shutil.copy2(exported_path, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a YOLO checkpoint to ONNX")
    parser.add_argument("--weights", required=True,
                        help="Path to the trained .pt weights")
    parser.add_argument("--output", default=None,
                        help="Destination .onnx file")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    onnx_path = export_onnx_model(
        args.weights, args.output, imgsz=320, opset=args.opset)
    print(onnx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
