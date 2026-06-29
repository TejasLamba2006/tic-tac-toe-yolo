"""Base class for pipeline stages.

Every stage inherits from ``Stage`` and implements ``run()``.
The runner calls ``should_skip()`` first to support resume behaviour.
"""

from __future__ import annotations

import enum
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import PipelineContext


class StageStatus(enum.Enum):
    """Outcome of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageResult:
    """Structured result returned by every stage."""

    status: StageStatus
    message: str = ""
    duration_seconds: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (StageStatus.SUCCESS, StageStatus.SKIPPED)


class Stage(ABC):
    """Abstract base for a single pipeline stage."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"pipeline.stages.{name}")

    @abstractmethod
    def run(self, ctx: PipelineContext) -> StageResult:
        """Execute the stage.  Must return a ``StageResult``."""

    def should_skip(self, ctx: PipelineContext) -> str | None:
        """Return a reason string if this stage should be skipped, else ``None``.

        Override in subclasses to implement resume / skip logic.
        """
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        """Return a list of error messages if required inputs are missing.

        Override in subclasses.  An empty list means inputs are valid.
        """
        return []

    def execute(self, ctx: PipelineContext) -> StageResult:
        """Orchestration wrapper: validate, skip-check, run, and time."""
        # Input validation
        errors = self.validate_inputs(ctx)
        if errors:
            msg = f"Input validation failed: {'; '.join(errors)}"
            self.logger.error(msg)
            return StageResult(status=StageStatus.FAILED, message=msg)

        # Skip check
        skip_reason = self.should_skip(ctx)
        if skip_reason is not None:
            self.logger.info("Skipping: %s", skip_reason)
            return StageResult(
                status=StageStatus.SKIPPED, message=skip_reason
            )

        # Run
        self.logger.info("Starting stage: %s", self.name)
        start = time.perf_counter()
        try:
            result = self.run(ctx)
        except Exception as exc:
            duration = time.perf_counter() - start
            self.logger.exception("Stage failed with exception")
            return StageResult(
                status=StageStatus.FAILED,
                message=str(exc),
                duration_seconds=duration,
            )
        result.duration_seconds = time.perf_counter() - start
        self.logger.info(
            "Completed: %s (%.1fs)", result.status.value, result.duration_seconds
        )
        return result
