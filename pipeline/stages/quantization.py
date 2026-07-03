"""Stage 6: TFLite INT8 quantization.

Reuses the quantization logic from ``scripts/tflite_quant.py`` but replaces
the Hydra configuration loader with plain YAML/dataclass configuration.

The existing script used Hydra solely for config loading (``@hydra.main``),
which pulled in hydra, omegaconf, and munch as dependencies.  This stage
achieves the same result with zero extra dependencies beyond TensorFlow.

Workflow:
    1. Generate a TensorFlow SavedModel from best.pt (via Ultralytics export)
    2. Quantize the SavedModel to INT8 TFLite using a representative dataset
    3. Save the quantized .tflite to the artifacts directory
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class QuantizationStage(Stage):
    """Quantize the model to INT8 TFLite."""

    def __init__(self) -> None:
        super().__init__("quantization")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if not ctx.config.quantization.enabled:
            return "Quantization is disabled in config"
        output_dir = ctx.stage_artifacts_dir(self.name)
        # Check for any .tflite file
        tflite_files = list(output_dir.glob("*.tflite"))
        if tflite_files:
            ctx.tflite_model = tflite_files[0]
            return f"Quantized model already exists: {tflite_files[0]}"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.config.quantization.enabled:
            return errors

        if ctx.best_weights is None or not ctx.best_weights.is_file():
            errors.append(
                "Trained weights (best.pt) not found. "
                "Run the training stage first."
            )

        # Check calibration dataset
        calib_path = self._resolve_calib_path(ctx)
        if calib_path is None or not calib_path.is_dir():
            errors.append(
                "Calibration dataset not found. Set "
                "quantization.calib_dataset_path or ensure dataset images exist."
            )

        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        try:
            import tensorflow as tf
        except ImportError as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"TensorFlow is required for quantization: {exc}",
            )

        cfg = ctx.config.quantization
        output_dir = ctx.stage_artifacts_dir(self.name)

        # Step 1: Generate SavedModel from best.pt
        saved_model_dir = self._generate_saved_model(ctx, output_dir)
        if saved_model_dir is None:
            return StageResult(
                status=StageStatus.FAILED,
                message="Failed to generate SavedModel for quantization",
            )

        # Step 2: Set up seeds for reproducibility
        self._setup_seeds(42)

        # Step 3: Configure TFLite converter
        self.logger.info("Loading SavedModel from %s", saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

        # Build representative dataset generator
        calib_path = self._resolve_calib_path(ctx)
        input_shape = cfg.input_shape

        def representative_data_gen():
            import cv2
            if cfg.fake:
                for _ in range(5):
                    data = np.random.rand(1, *input_shape)
                    yield [data.astype(np.float32)]
            else:
                image_files = [
                    f for f in os.listdir(str(calib_path))
                    if f.lower().endswith(".jpg")
                ]
                for image_file in image_files:
                    image = cv2.imread(
                        os.path.join(str(calib_path), image_file)
                    )
                    if image is None:
                        continue
                    if len(image.shape) != 3:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(
                        image,
                        (int(input_shape[0]), int(input_shape[1])),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    normalized = (
                        resized / cfg.rescaling_scale + cfg.rescaling_offset
                    )
                    processed = np.expand_dims(
                        normalized.astype(np.float32), 0
                    )
                    yield [processed]

        # Configure quantization type
        quant_tag = "quant_pc"
        if cfg.quantization_type == "per_tensor":
            converter._experimental_disable_per_channel = True
            quant_tag = "quant_pt"

        # Configure input type
        input_tag = "f"
        if cfg.quantization_input_type == "int8":
            converter.inference_input_type = tf.int8
            input_tag = "i"
        elif cfg.quantization_input_type == "uint8":
            converter.inference_input_type = tf.uint8
            input_tag = "u"

        # Configure output type
        output_tag = "f"
        if cfg.quantization_output_type == "int8":
            converter.inference_output_type = tf.int8
            output_tag = "i"
        elif cfg.quantization_output_type == "uint8":
            converter.inference_output_type = tf.uint8
            output_tag = "u"

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen

        # Step 4: Convert
        self.logger.info(
            "Quantizing: type=%s, input=%s, output=%s",
            cfg.quantization_type,
            cfg.quantization_input_type,
            cfg.quantization_output_type,
        )
        tflite_model = converter.convert()

        # Step 5: Save with naming convention from existing scripts
        use_case = cfg.use_case if not cfg.fake else "fake"
        filename = (
            f"{cfg.model_name}_{quant_tag}_{input_tag}{output_tag}_{use_case}.tflite"
        )
        tflite_path = output_dir / filename
        tflite_path.write_bytes(tflite_model)

        ctx.tflite_model = tflite_path

        msg = f"Quantized model: {tflite_path.name} ({len(tflite_model)} bytes)"
        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={"tflite_model": str(tflite_path)},
            metrics={
                "model_size_bytes": len(tflite_model),
                "quantization_type": cfg.quantization_type,
                "input_type": cfg.quantization_input_type,
                "output_type": cfg.quantization_output_type,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_calib_path(self, ctx: PipelineContext) -> Path | None:
        """Resolve the calibration dataset path."""
        cfg = ctx.config.quantization
        if cfg.calib_dataset_path:
            p = Path(cfg.calib_dataset_path)
            if p.is_absolute():
                return p
            return ctx.project_root / p

        # Auto-resolve from dataset config
        ds = ctx.config.dataset
        candidates = [
            ctx.dataset_dir / ds.train_images,
            ctx.dataset_dir / "images",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def _generate_saved_model(
        self, ctx: PipelineContext, output_dir: Path
    ) -> Path | None:
        """Generate a TensorFlow SavedModel from best.pt.

        Strategy (avoids onnx2tf pickle bug with numpy >=2.x):
        1. Export ONNX via Ultralytics (works fine).
        2. Convert ONNX → SavedModel via a subprocess wrapper that
           patches ``np.load`` before importing onnx2tf.
        3. The TFLite INT8 quantization is then performed in ``run()``
           via ``tf.lite.TFLiteConverter``.
        """
        saved_model_dir = output_dir / "saved_model"
        if saved_model_dir.is_dir() and (saved_model_dir / "saved_model.pb").exists():
            self.logger.info("Reusing existing SavedModel: %s", saved_model_dir)
            ctx.saved_model_dir = saved_model_dir
            return saved_model_dir

        onnx_path = output_dir / "best.onnx"

        # Step 1: Ensure we have an ONNX file.
        if not onnx_path.exists():
            try:
                from ultralytics import YOLO
            except ImportError:
                self.logger.error("Ultralytics is required for ONNX export")
                return None

            self.logger.info("Exporting ONNX from %s", ctx.best_weights)
            model = YOLO(str(ctx.best_weights))
            try:
                exported = model.export(
                    format="onnx",
                    imgsz=ctx.config.training.imgsz,
                    opset=12,
                )
                exported_path = Path(str(exported))
                if exported_path != onnx_path:
                    import shutil
                    shutil.copy2(str(exported_path), str(onnx_path))
            except Exception as exc:
                self.logger.error("ONNX export failed: %s", exc)
                return None

        # Step 2: Convert ONNX → SavedModel using onnx2tf CLI in a subprocess.
        # This avoids the numpy >=2.x pickle incompatibility that occurs
        # when onnx2tf is imported in-process.
        export_dir = output_dir / "onnx2tf_export"
        if export_dir.exists():
            import shutil
            shutil.rmtree(export_dir)

        self.logger.info("Converting ONNX → SavedModel via onnx2tf CLI: %s", onnx_path)

        # Write a small wrapper script that patches np.load before onnx2tf.
        # Use a .tmp extension so uvicorn's watchfiles reloader never treats
        # it as Python source and reloads the server mid-conversion.
        wrapper_script = output_dir / "_onnx2tf_convert.tmp"
        wrapper_script.write_text(
            "import sys, os, io, numpy as np\n"
            "\n"
            "# Patch: return dummy calibration data instead of downloading\n"
            "# from GitHub (the onnx2tf test data URL returns 404).\n"
            "def _dummy_download():\n"
            "    return np.random.rand(20, 128, 128, 3).astype(np.float32)\n"
            "\n"
            "import onnx2tf.utils.common_functions as cf\n"
            "cf.download_test_image_data = _dummy_download\n"
            "import onnx2tf.onnx2tf as o2t\n"
            "o2t.download_test_image_data = _dummy_download\n"
            "\n"
            "import onnx2tf\n"
            f"onnx2tf.convert(\n"
            f"    input_onnx_file_path=r\"{onnx_path}\",\n"
            f"    output_folder_path=r\"{export_dir}\",\n"
            "    non_verbose=True,\n"
            ")\n"
        )

        import subprocess
        import threading

        # On Windows, CREATE_NEW_PROCESS_GROUP detaches the child from the
        # parent's console process group so that Ctrl+C / SIGINT events
        # (e.g. from uvicorn's terminal) are NOT forwarded to this process.
        _popen_kwargs: dict = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr → stdout for single stream
            cwd=str(output_dir),
        )
        if sys.platform == "win32":
            _popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        def _stream_to_logger(pipe) -> list[str]:
            """Read lines from *pipe* and forward to self.logger in real time.
            Returns accumulated lines so we can surface errors after the fact.
            """
            lines: list[str] = []
            try:
                for raw in iter(pipe.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        lines.append(line)
                        self.logger.info("[onnx2tf] %s", line)
            except Exception:
                pass
            return lines


        try:
            _proc = subprocess.Popen(
                [sys.executable, str(wrapper_script)],
                **_popen_kwargs,
            )

            # Stream output in a daemon thread so it appears live in the UI.
            _log_lines: list[list[str]] = [[]]
            _reader = threading.Thread(
                target=lambda: _log_lines.__setitem__(0, _stream_to_logger(_proc.stdout)),
                daemon=True,
            )
            _reader.start()

            try:
                _proc.wait(timeout=300)
                _reader.join(timeout=5)
            except KeyboardInterrupt:
                self.logger.warning(
                    "onnx2tf conversion interrupted — terminating child process"
                )
                _proc.terminate()
                try:
                    _proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _proc.kill()
                raise
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait()
                self.logger.error("onnx2tf conversion timed out after 300 s")
                return None
        except KeyboardInterrupt:
            raise

        returncode = _proc.returncode
        if returncode != 0:
            tail = "\n".join(_log_lines[0][-40:]) if _log_lines[0] else "(no output)"
            self.logger.error("onnx2tf CLI failed (exit %d):\n%s", returncode, tail)
            return None

        # Step 3: Locate saved_model.pb in the exported tree.
        import shutil

        for candidate in [export_dir, export_dir.parent]:
            if (candidate / "saved_model.pb").exists():
                if candidate.resolve() != saved_model_dir.resolve():
                    if saved_model_dir.exists():
                        shutil.rmtree(saved_model_dir)
                    shutil.copytree(candidate, saved_model_dir)
                ctx.saved_model_dir = saved_model_dir
                return saved_model_dir

        for pb in export_dir.rglob("saved_model.pb"):
            sm_dir = pb.parent
            if sm_dir.resolve() != saved_model_dir.resolve():
                if saved_model_dir.exists():
                    shutil.rmtree(saved_model_dir)
                shutil.copytree(sm_dir, saved_model_dir)
            ctx.saved_model_dir = saved_model_dir
            return saved_model_dir

        self.logger.error(
            "Could not locate saved_model.pb after onnx2tf export. "
            "Exported path: %s  Contents: %s",
            export_dir,
            list(export_dir.iterdir()) if export_dir.is_dir() else "N/A",
        )
        return None

    def _setup_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
