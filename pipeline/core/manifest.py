"""Pipeline manifest: persistent run state saved to YAML.

The manifest is the single source of truth for every pipeline run.
It records:
- Unique run ID (timestamp-based)
- Per-stage status, timestamps, duration, and messages
- All artifact paths (best.pt, .onnx, .tflite, .nb, reports)
- Key metrics (mAP50, precision, recall, F1, model size, latency)
- Configuration snapshot

The manifest is saved to ``artifacts/pipeline_run.yaml`` after every
stage completes, making it crash-safe and resumable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .stage import StageResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _generate_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


class PipelineManifest:
    """Persistent pipeline run state."""

    def __init__(self, artifacts_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or _generate_run_id()
        self.started_at = _now_iso()
        self.completed_at: str | None = None
        self.overall_status = "running"
        self.config_snapshot: dict[str, Any] = {}
        self.stages: dict[str, dict[str, Any]] = {}
        self.artifacts: dict[str, str] = {}
        self.metrics: dict[str, Any] = {}
        self._path = artifacts_dir / "pipeline_run.yaml"

    @classmethod
    def load(cls, artifacts_dir: Path) -> PipelineManifest | None:
        """Load an existing manifest from disk, or return ``None``."""
        path = artifacts_dir / "pipeline_run.yaml"
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                return None
            m = cls(artifacts_dir, run_id=data.get("run_id"))
            m.started_at = data.get("started_at", m.started_at)
            m.completed_at = data.get("completed_at")
            m.overall_status = data.get("status", "unknown")
            m.config_snapshot = data.get("config", {})
            m.stages = data.get("stages", {})
            m.artifacts = data.get("artifacts", {})
            m.metrics = data.get("metrics", {})
            logger.info("Loaded manifest: run_id=%s", m.run_id)
            return m
        except Exception:
            logger.warning("Failed to load manifest from %s", path, exc_info=True)
            return None

    def set_config_snapshot(self, config: Any) -> None:
        """Store a lightweight config summary in the manifest."""
        self.config_snapshot = {
            "model": getattr(config.training, "model", ""),
            "epochs": getattr(config.training, "epochs", 0),
            "imgsz": getattr(config.training, "imgsz", 0),
            "batch": getattr(config.training, "batch", 0),
            "dataset": getattr(config.dataset, "path", ""),
            "quantization_type": getattr(
                config.quantization, "quantization_type", ""
            ),
        }

    def update_stage(self, stage_name: str, result: StageResult) -> None:
        """Record the result of a completed stage."""
        self.stages[stage_name] = {
            "status": result.status.value,
            "message": result.message,
            "duration_seconds": round(result.duration_seconds, 2),
            "completed_at": _now_iso(),
        }

        # Merge artifact paths into the top-level artifacts dict.
        for key, value in result.artifacts.items():
            self.artifacts[key] = value

        # Merge numeric metrics into the top-level metrics dict.
        for key, value in result.metrics.items():
            if isinstance(value, (int, float, str)):
                self.metrics[key] = value

    def update_artifact(self, key: str, path: str) -> None:
        """Record an artifact path."""
        self.artifacts[key] = path

    def finalize(self, overall_status: str = "success") -> None:
        """Mark the pipeline run as complete."""
        self.completed_at = _now_iso()
        self.overall_status = overall_status

    def save(self) -> Path:
        """Write the manifest to disk and return the file path."""
        data = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.overall_status,
            "config": self.config_snapshot,
            "stages": self.stages,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
        logger.debug("Manifest saved: %s", self._path)
        return self._path

    def restore_context_artifacts(self, ctx: Any) -> None:
        """Restore artifact paths from the manifest into a PipelineContext.

        Used when resuming a pipeline run so downstream stages can find
        artifacts produced by earlier stages.
        """
        from pathlib import Path as _Path

        mapping = {
            "best_weights": "best_weights",
            "onnx_model": "onnx_model",
            "tflite_model": "tflite_model",
            "nb_model": "nb_model",
        }
        for manifest_key, ctx_attr in mapping.items():
            value = self.artifacts.get(manifest_key)
            if value and _Path(value).is_file():
                setattr(ctx, ctx_attr, _Path(value))
                logger.debug("Restored %s = %s", ctx_attr, value)
