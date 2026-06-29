"""Stage 3: YOLO training.

Wraps the Ultralytics YOLO training API with full configurability.
Automatically saves best.pt, last.pt, and training metrics.

NOTE: The existing ``scripts/train.py`` passed ``format="onnx"`` and
``opset=12`` to ``model.train()``.  These are export parameters that
Ultralytics silently ignores during training.  This stage correctly
separates training from export.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class TrainingStage(Stage):
    """Train a YOLO model using Ultralytics."""

    def __init__(self) -> None:
        super().__init__("training")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        output_dir = ctx.stage_artifacts_dir(self.name)
        best = output_dir / "weights" / "best.pt"
        if best.is_file():
            ctx.best_weights = best
            return f"Trained weights already exist: {best}"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        cfg = ctx.config.training

        model_path = ctx.project_root / cfg.model
        if not model_path.is_file():
            errors.append(f"Base model not found: {model_path}")

        data_yaml = ctx.project_root / cfg.data_yaml
        if not data_yaml.is_file():
            errors.append(f"Data YAML not found: {data_yaml}")

        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Ultralytics is required for training: {exc}",
            )

        cfg = ctx.config.training
        output_dir = ctx.stage_artifacts_dir(self.name)

        model_path = str(ctx.project_root / cfg.model)
        data_yaml = str(ctx.project_root / cfg.data_yaml)

        self.logger.info("Loading base model: %s", model_path)
        model = YOLO(model_path)

        # Build training kwargs from config
        train_kwargs = {
            "data": data_yaml,
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "workers": cfg.workers,
            "patience": cfg.patience,
            "optimizer": cfg.optimizer,
            "lr0": cfg.lr0,
            "lrf": cfg.lrf,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
            "amp": cfg.amp,
            "cache": cfg.cache,
            "plots": cfg.plots,
            "save": cfg.save,
            "exist_ok": cfg.exist_ok,
            "resume": cfg.resume,
            "project": str(output_dir),
            "name": cfg.name or "train",
        }
        if cfg.device is not None:
            train_kwargs["device"] = cfg.device

        self.logger.info(
            "Starting training: %d epochs, batch=%d, imgsz=%d",
            cfg.epochs,
            cfg.batch,
            cfg.imgsz,
        )
        results = model.train(**train_kwargs)

        # Locate output weights
        train_dir = output_dir / (cfg.name or "train")
        weights_dir = train_dir / "weights"

        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"

        if not best_pt.is_file():
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Training completed but best.pt not found at {best_pt}",
            )

        # Also copy weights to a stable location for downstream stages
        stable_weights_dir = output_dir / "weights"
        stable_weights_dir.mkdir(parents=True, exist_ok=True)
        stable_best = stable_weights_dir / "best.pt"
        shutil.copy2(best_pt, stable_best)
        ctx.best_weights = stable_best

        if last_pt.is_file():
            stable_last = stable_weights_dir / "last.pt"
            shutil.copy2(last_pt, stable_last)
            ctx.last_weights = stable_last

        # Extract metrics if available
        metrics: dict = {}
        try:
            if results is not None:
                results_dict = getattr(results, "results_dict", None)
                if results_dict:
                    metrics = {k: float(v) for k, v in results_dict.items()}
        except Exception:
            self.logger.debug("Could not extract training metrics", exc_info=True)

        ctx.training_metrics = metrics

        # Save metrics to JSON
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        msg = (
            f"Training complete. best.pt saved to {stable_best}"
        )
        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={
                "best_weights": str(stable_best),
                "train_dir": str(train_dir),
                "metrics": str(metrics_path),
            },
            metrics=metrics,
        )
