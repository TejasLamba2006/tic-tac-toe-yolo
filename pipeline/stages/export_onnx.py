"""Stage 5: Export trained model to ONNX.

Reuses the export logic from ``src/stm32/export_to_onnx.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class ExportOnnxStage(Stage):
    """Export best.pt to ONNX format."""

    def __init__(self) -> None:
        super().__init__("export")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        output_dir = ctx.stage_artifacts_dir(self.name)
        onnx_path = output_dir / "best.onnx"
        if onnx_path.is_file():
            ctx.onnx_model = onnx_path
            return f"ONNX model already exists: {onnx_path}"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if ctx.best_weights is None or not ctx.best_weights.is_file():
            errors.append(
                "Trained weights (best.pt) not found. "
                "Run the training stage first."
            )
        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        # Import the existing export function from the project
        try:
            from src.stm32.export_to_onnx import export_onnx_model
        except ImportError:
            # Fallback: use Ultralytics directly
            return self._export_direct(ctx)

        cfg = ctx.config.export
        output_dir = ctx.stage_artifacts_dir(self.name)
        onnx_path = output_dir / "best.onnx"

        self.logger.info(
            "Exporting %s to ONNX (opset=%d, imgsz=%d)",
            ctx.best_weights,
            cfg.opset,
            cfg.imgsz,
        )

        try:
            result_path = export_onnx_model(
                weights_path=ctx.best_weights,
                output_path=onnx_path,
                imgsz=cfg.imgsz,
                opset=cfg.opset,
            )
        except Exception as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"ONNX export failed: {exc}",
            )

        ctx.onnx_model = Path(result_path)

        msg = f"ONNX model exported: {ctx.onnx_model}"
        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={"onnx_model": str(ctx.onnx_model)},
        )

    def _export_direct(self, ctx: PipelineContext) -> StageResult:
        """Fallback export using Ultralytics directly."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Ultralytics is required for ONNX export: {exc}",
            )

        import shutil

        cfg = ctx.config.export
        output_dir = ctx.stage_artifacts_dir(self.name)
        onnx_path = output_dir / "best.onnx"

        self.logger.info("Exporting via Ultralytics (src.stm32 not available)")
        model = YOLO(str(ctx.best_weights))
        exported = model.export(
            format="onnx",
            imgsz=cfg.imgsz,
            opset=cfg.opset,
            simplify=cfg.simplify,
        )
        exported_path = Path(str(exported))
        if exported_path.resolve() != onnx_path.resolve():
            shutil.copy2(exported_path, onnx_path)

        ctx.onnx_model = onnx_path
        msg = f"ONNX model exported: {onnx_path}"
        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={"onnx_model": str(onnx_path)},
        )
