from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Sequence


def _export_from_pt(weights_path: Path, destination: Path, imgsz: int) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Ultralytics is required to export TFLite models from PyTorch weights") from exc

    model = YOLO(str(weights_path))
    exported = model.export(format="tflite", imgsz=imgsz, simplify=True, int8=True, data="data.yaml")
    exported_path = Path(str(exported))

    if exported_path.is_dir():
        # Find the .tflite file inside the directory
        tflite_files = list(exported_path.glob("*.tflite"))
        if not tflite_files:
            # Check nested directories if any
            tflite_files = list(exported_path.glob("**/*.tflite"))
        if not tflite_files:
            raise FileNotFoundError(
                f"No .tflite file found in the exported folder: {exported_path}"
            )
        source_file = tflite_files[0]
    else:
        source_file = exported_path

    if source_file.resolve() != destination.resolve():
        shutil.copy2(source_file, destination)
    return destination


def _export_from_onnx(onnx_path: Path, destination: Path) -> Path:
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "ONNX -> TFLite conversion needs onnx, onnx-tf, and tensorflow installed"
        ) from exc

    model = onnx.load(str(onnx_path))
    tf_rep = prepare(model)

    with tempfile.TemporaryDirectory() as temp_dir:
        saved_model_dir = Path(temp_dir) / "saved_model"
        tf_rep.export_graph(str(saved_model_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(saved_model_dir))
        converter.optimizations = []
        tflite_model = converter.convert()

    destination.write_bytes(tflite_model)
    return destination


def export_tflite_model(
    source_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
) -> Path:
    """Export a model to TFLite.

    If ``source_path`` ends in ``.onnx``, the script performs ONNX -> SavedModel ->
    TFLite conversion. If it ends in ``.pt``, Ultralytics handles the export.
    """

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source model not found: {source}")

    destination = Path(
        output_path) if output_path is not None else source.with_suffix(".tflite")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".pt":
        return _export_from_pt(source, destination, imgsz)
    if source.suffix.lower() == ".onnx":
        return _export_from_onnx(source, destination)

    raise ValueError(f"Unsupported source model format: {source.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a model to TFLite")
    parser.add_argument("--source", required=True,
                        help="Path to the source model (.onnx or .pt)")
    parser.add_argument("--output", default=None,
                        help="Destination .tflite file")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Export image size for PyTorch sources")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tflite_path = export_tflite_model(
        args.source, args.output, imgsz=args.imgsz)
    print(tflite_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
