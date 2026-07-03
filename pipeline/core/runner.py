"""Pipeline runner: orchestrates stage execution in order.

Supports:
- Full pipeline execution
- Single-stage execution (``--stage``)
- Dry-run mode (``--dry-run``)
- Resume from a specific stage (``--from-stage``)
- Pipeline manifest persistence after every stage
"""

from __future__ import annotations

import logging
from typing import Sequence

from .context import PipelineContext
from .manifest import PipelineManifest
from .stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Execute an ordered sequence of ``Stage`` instances."""

    def __init__(self, stages: Sequence[Stage]) -> None:
        self.stages = list(stages)
        self._stage_map = {s.name: s for s in self.stages}

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def run_all(
        self,
        ctx: PipelineContext,
        *,
        from_stage: str | None = None,
        single_stage: str | None = None,
    ) -> dict[str, StageResult]:
        """Run stages and return a mapping of stage name to result.

        Parameters
        ----------
        ctx:
            Shared pipeline context.
        from_stage:
            If set, skip all stages before this one.
        single_stage:
            If set, run only this stage.
        """
        # Initialize or load the manifest.
        self._init_manifest(ctx, from_stage)

        results: dict[str, StageResult] = {}

        if single_stage:
            results = self._run_single(ctx, single_stage, results)
            self._finalize_manifest(ctx, results)
            return results

        started = from_stage is None
        for stage in self.stages:
            if not started:
                if stage.name == from_stage:
                    started = True
                else:
                    logger.info(
                        "Skipping stage '%s' (resuming from '%s')",
                        stage.name,
                        from_stage,
                    )
                    results[stage.name] = StageResult(
                        status=StageStatus.SKIPPED,
                        message=f"Skipped (resuming from {from_stage})",
                    )
                    continue

            if ctx.dry_run:
                result = self._dry_run_stage(stage, ctx)
            else:
                try:
                    result = stage.execute(ctx)
                except KeyboardInterrupt:
                    results[stage.name] = StageResult(
                        status=StageStatus.FAILED,
                        message="Interrupted",
                    )
                    ctx.record_result(stage.name, results[stage.name])
                    self._finalize_manifest(ctx, results)
                    raise

            results[stage.name] = result
            ctx.record_result(stage.name, result)

            if not result.ok:
                _RED = "\033[31m"
                _R = "\033[0m"
                _BOLD = "\033[1m"
                logger.error(
                    f"{_RED}{_BOLD}Stage '%s' failed{_R}{_RED}: %s{_R} "
                    f"— stopping pipeline.",
                    stage.name,
                    result.message,
                )
                break

        self._finalize_manifest(ctx, results)
        self._print_summary(results)
        return results

    def _init_manifest(
        self, ctx: PipelineContext, from_stage: str | None
    ) -> None:
        """Create a new manifest or load an existing one for resume."""
        if from_stage is not None:
            # Try to load existing manifest for resume.
            existing = PipelineManifest.load(ctx.artifacts_dir)
            if existing is not None:
                ctx.manifest = existing
                existing.restore_context_artifacts(ctx)
                logger.info(
                    "Resuming from manifest run_id=%s", existing.run_id
                )
                return

        manifest = PipelineManifest(ctx.artifacts_dir)
        manifest.set_config_snapshot(ctx.config)
        ctx.manifest = manifest
        if not ctx.dry_run:
            manifest.save()

    def _finalize_manifest(
        self,
        ctx: PipelineContext,
        results: dict[str, StageResult],
    ) -> None:
        """Mark the manifest as complete and save."""
        if ctx.manifest is None or ctx.dry_run:
            return

        # Determine overall status.
        failed = any(
            r.status == StageStatus.FAILED for r in results.values()
        )
        all_ok = all(r.ok for r in results.values())

        if failed:
            ctx.manifest.finalize("failed")
        elif all_ok:
            ctx.manifest.finalize("success")
        else:
            ctx.manifest.finalize("partial")

        # Record final artifact paths from context.
        if ctx.best_weights:
            ctx.manifest.update_artifact("best_weights", str(ctx.best_weights))
        if ctx.onnx_model:
            ctx.manifest.update_artifact("onnx_model", str(ctx.onnx_model))
        if ctx.tflite_model:
            ctx.manifest.update_artifact("tflite_model", str(ctx.tflite_model))
        if ctx.nb_model:
            ctx.manifest.update_artifact("nb_model", str(ctx.nb_model))

        path = ctx.manifest.save()
        logger.info("Manifest saved: %s", path)

    def _run_single(
        self,
        ctx: PipelineContext,
        stage_name: str,
        results: dict[str, StageResult],
    ) -> dict[str, StageResult]:
        stage = self._stage_map.get(stage_name)
        if stage is None:
            available = ", ".join(self.stage_names)
            msg = f"Unknown stage '{stage_name}'. Available: {available}"
            logger.error(msg)
            results[stage_name] = StageResult(
                status=StageStatus.FAILED, message=msg
            )
            return results

        if ctx.dry_run:
            result = self._dry_run_stage(stage, ctx)
        else:
            result = stage.execute(ctx)

        results[stage_name] = result
        ctx.record_result(stage_name, result)
        self._print_summary(results)
        return results

    def _dry_run_stage(
        self, stage: Stage, ctx: PipelineContext
    ) -> StageResult:
        """Validate inputs and report what would happen without executing."""
        errors = stage.validate_inputs(ctx)
        skip = stage.should_skip(ctx)

        if errors:
            msg = f"[DRY RUN] Would FAIL: {'; '.join(errors)}"
            logger.warning(msg)
            return StageResult(status=StageStatus.FAILED, message=msg)

        if skip:
            msg = f"[DRY RUN] Would SKIP: {skip}"
            logger.info(msg)
            return StageResult(status=StageStatus.SKIPPED, message=msg)

        msg = f"[DRY RUN] Would RUN stage '{stage.name}'"
        logger.info(msg)
        return StageResult(status=StageStatus.SUCCESS, message=msg)

    def _print_summary(self, results: dict[str, StageResult]) -> None:
        # ANSI helpers — same palette as _ColorFormatter but applied via
        # the plain logger so these messages bypass the formatter.
        _R = "\033[0m"
        _CYAN = "\033[36m"
        _BOLD = "\033[1m"
        _RED = "\033[31m"
        _GREEN = "\033[32m"
        _YELLOW = "\033[33m"
        _DIM = "\033[2m"

        _ICONS = {
            StageStatus.SUCCESS: (_GREEN, "OK"),
            StageStatus.SKIPPED: (_YELLOW, "SKIP"),
            StageStatus.FAILED:  (_RED, "FAIL"),
            StageStatus.PENDING: (_DIM, "----"),
            StageStatus.RUNNING: (_DIM, "...."),
        }

        logger.info("")
        logger.info(f"{_CYAN}{_BOLD}{'=' * 60}{_R}")
        logger.info(f"{_CYAN}{_BOLD}  PIPELINE SUMMARY{_R}")
        logger.info(f"{_CYAN}{_BOLD}{'=' * 60}{_R}")
        for name, result in results.items():
            color, icon = _ICONS.get(result.status, (_DIM, "????"))
            duration = (
                f" ({result.duration_seconds:.1f}s)"
                if result.duration_seconds > 0
                else ""
            )
            logger.info(
                f"  {color}[{icon}]{_R} {_DIM}{name:<20s}{_R} "
                f"{result.message}{_DIM}{duration}{_R}"
            )
        logger.info(f"{_CYAN}{_BOLD}{'=' * 60}{_R}")
