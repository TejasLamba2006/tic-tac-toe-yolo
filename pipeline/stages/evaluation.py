"""Stage 4: Model evaluation.

Runs YOLO validation on the trained model and collects:
- Precision, Recall, mAP@50, mAP@50-95
- F1 score (computed from P and R)
- Per-class metrics
- Confusion matrix (if plots are enabled)

Results are stored as JSON in the stage artifacts directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class EvaluationStage(Stage):
    """Evaluate the trained YOLO model."""

    def __init__(self) -> None:
        super().__init__("evaluation")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        report = ctx.stage_artifacts_dir(self.name) / "evaluation_report.json"
        if report.is_file():
            return f"Evaluation report already exists: {report}"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if ctx.best_weights is None or not ctx.best_weights.is_file():
            errors.append(
                "Trained weights (best.pt) not found. "
                "Run the training stage first."
            )
        data_yaml = ctx.project_root / ctx.config.training.data_yaml
        if not data_yaml.is_file():
            errors.append(f"Data YAML not found: {data_yaml}")
        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Ultralytics is required for evaluation: {exc}",
            )

        output_dir = ctx.stage_artifacts_dir(self.name)
        data_yaml = str(ctx.project_root / ctx.config.training.data_yaml)

        self.logger.info("Loading model: %s", ctx.best_weights)
        model = YOLO(str(ctx.best_weights))

        self.logger.info("Running validation...")
        results = model.val(
            data=data_yaml,
            imgsz=ctx.config.training.imgsz,
            batch=ctx.config.training.batch,
            workers=ctx.config.training.workers,
            plots=True,
            project=str(output_dir),
            name="val",
            exist_ok=True,
        )

        # Extract metrics
        metrics: dict = {}
        try:
            box = getattr(results, "box", None)
            if box is not None:
                metrics["precision"] = float(box.mp)
                metrics["recall"] = float(box.mr)
                metrics["mAP50"] = float(box.map50)
                metrics["mAP50_95"] = float(box.map)

                p = metrics["precision"]
                r = metrics["recall"]
                metrics["f1"] = (
                    2 * p * r / (p + r) if (p + r) > 0 else 0.0
                )

                # Per-class metrics
                if hasattr(box, "ap_class_index") and hasattr(box, "p"):
                    class_names = getattr(results, "names", {})
                    per_class = {}
                    for i, cls_idx in enumerate(box.ap_class_index):
                        cls_name = class_names.get(int(cls_idx), str(cls_idx))
                        per_class[cls_name] = {
                            "precision": float(box.p[i]),
                            "recall": float(box.r[i]),
                            "ap50": float(box.ap50[i]),
                            "ap": float(box.ap[i]),
                        }
                    metrics["per_class"] = per_class
        except Exception:
            self.logger.warning("Could not extract all metrics", exc_info=True)

        # Try to extract from results_dict as fallback
        if not metrics:
            results_dict = getattr(results, "results_dict", None)
            if results_dict:
                metrics = {k: float(v) for k, v in results_dict.items()}

        # Save report
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        # Copy confusion matrix if it exists
        val_dir = output_dir / "val"
        if val_dir.is_dir():
            for plot_file in val_dir.glob("*.png"):
                shutil.copy2(plot_file, output_dir / plot_file.name)

        msg_parts = []
        for key in ("precision", "recall", "mAP50", "mAP50_95", "f1"):
            if key in metrics:
                msg_parts.append(f"{key}={metrics[key]:.4f}")
        msg = "Evaluation: " + ", ".join(msg_parts) if msg_parts else "Evaluation complete"

        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={"report": str(report_path)},
            metrics=metrics,
        )
