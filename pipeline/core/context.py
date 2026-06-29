"""Pipeline context: shared state passed between stages.

The context holds the resolved configuration, artifact paths, the
pipeline manifest, and any intermediate results that downstream
stages need.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.schema import PipelineConfig
from ..utils.paths import ensure_directory, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Mutable state shared across all pipeline stages."""

    config: PipelineConfig
    project_root: Path
    dry_run: bool = False

    # Paths resolved at init time.
    artifacts_dir: Path = field(init=False)
    dataset_dir: Path = field(init=False)

    # Pipeline manifest (set by the runner after construction).
    manifest: Any = field(default=None, init=False)

    # Intermediate artifacts populated by stages.
    # Stages write paths here so downstream stages can find them.
    best_weights: Path | None = field(default=None, init=False)
    last_weights: Path | None = field(default=None, init=False)
    onnx_model: Path | None = field(default=None, init=False)
    saved_model_dir: Path | None = field(default=None, init=False)
    tflite_model: Path | None = field(default=None, init=False)
    nb_model: Path | None = field(default=None, init=False)
    training_metrics: dict[str, Any] = field(default_factory=dict, init=False)
    stage_results: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.artifacts_dir = resolve_path(
            self.config.artifacts_dir, self.project_root
        )
        self.dataset_dir = resolve_path(
            self.config.dataset.path, self.project_root
        )

        if not self.dry_run:
            ensure_directory(self.artifacts_dir)

        logger.info("Project root:   %s", self.project_root)
        logger.info("Artifacts dir:  %s", self.artifacts_dir)
        logger.info("Dataset dir:    %s", self.dataset_dir)

    def stage_artifacts_dir(self, stage_name: str) -> Path:
        """Return (and create) the artifacts subdirectory for a stage."""
        d = self.artifacts_dir / stage_name
        if not self.dry_run:
            ensure_directory(d)
        return d

    def record_result(self, stage_name: str, result: Any) -> None:
        """Store a stage result for later reporting and update the manifest."""
        self.stage_results[stage_name] = result
        if self.manifest is not None and hasattr(result, "status"):
            self.manifest.update_stage(stage_name, result)
            self.manifest.save()
