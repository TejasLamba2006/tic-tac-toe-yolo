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

        This replicates what ``scripts/export_saved.py`` does:
        call ``model.export(format="tflite", int8=True)`` which produces
        a SavedModel as an intermediate artifact.
        """
        saved_model_dir = output_dir / "saved_model"
        if saved_model_dir.is_dir() and (saved_model_dir / "saved_model.pb").exists():
            self.logger.info("Reusing existing SavedModel: %s", saved_model_dir)
            ctx.saved_model_dir = saved_model_dir
            return saved_model_dir

        try:
            from ultralytics import YOLO
        except ImportError:
            self.logger.error("Ultralytics is required to generate SavedModel")
            return None

        self.logger.info("Generating SavedModel from %s", ctx.best_weights)
        model = YOLO(str(ctx.best_weights))

        try:
            exported = model.export(
                format="tflite",
                imgsz=ctx.config.training.imgsz,
                int8=True,
                data=str(ctx.project_root / ctx.config.training.data_yaml),
            )
        except Exception as exc:
            self.logger.error("SavedModel export failed: %s", exc)
            return None

        exported_path = Path(str(exported))

        # Ultralytics may return the .tflite file or the SavedModel directory.
        # We need to find the saved_model directory.
        if exported_path.is_dir():
            candidate = exported_path
        else:
            candidate = exported_path.parent

        # Look for saved_model.pb
        if (candidate / "saved_model.pb").exists():
            import shutil
            if candidate.resolve() != saved_model_dir.resolve():
                if saved_model_dir.exists():
                    shutil.rmtree(saved_model_dir)
                shutil.copytree(candidate, saved_model_dir)
            ctx.saved_model_dir = saved_model_dir
            return saved_model_dir

        # Search parent directories
        for parent in [candidate.parent, candidate.parent.parent]:
            sm = parent / "saved_model"
            if sm.is_dir() and (sm / "saved_model.pb").exists():
                import shutil
                if sm.resolve() != saved_model_dir.resolve():
                    if saved_model_dir.exists():
                        shutil.rmtree(saved_model_dir)
                    shutil.copytree(sm, saved_model_dir)
                ctx.saved_model_dir = saved_model_dir
                return saved_model_dir

        # Try to find any saved_model.pb in the export tree
        for pb in candidate.rglob("saved_model.pb"):
            import shutil
            sm_dir = pb.parent
            if sm_dir.resolve() != saved_model_dir.resolve():
                if saved_model_dir.exists():
                    shutil.rmtree(saved_model_dir)
                shutil.copytree(sm_dir, saved_model_dir)
            ctx.saved_model_dir = saved_model_dir
            return saved_model_dir

        self.logger.error(
            "Could not locate SavedModel after export. "
            "Exported path: %s",
            exported_path,
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
