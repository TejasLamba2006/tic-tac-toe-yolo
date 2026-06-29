"""Command-line interface for the pipeline.

Provides the ``main()`` function used by both ``run_pipeline.py`` and
``python -m pipeline``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config.loader import load_config
from .core.context import PipelineContext
from .core.runner import PipelineRunner
from .core.stage import StageStatus
from .utils.logging import setup_logging
from .utils.paths import resolve_project_root

# Stage imports
from .stages.dataset_validation import DatasetValidationStage
from .stages.augmentation import AugmentationStage
from .stages.training import TrainingStage
from .stages.evaluation import EvaluationStage
from .stages.export_onnx import ExportOnnxStage
from .stages.quantization import QuantizationStage
from .stages.stedge_compile import STEdgeCompileStage
from .stages.deployment import DeploymentStage
from .stages.verification import VerificationStage
from .stages.benchmark import BenchmarkStage
from .stages.report import ReportStage


def _build_stages() -> list:
    """Instantiate all pipeline stages in execution order."""
    return [
        DatasetValidationStage(),
        AugmentationStage(),
        TrainingStage(),
        EvaluationStage(),
        ExportOnnxStage(),
        QuantizationStage(),
        STEdgeCompileStage(),
        DeploymentStage(),
        VerificationStage(),
        BenchmarkStage(),
        ReportStage(),
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Automated ML pipeline: dataset to deployment.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: pipeline/config/config.yaml)",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Run only this stage (e.g., training, export, quantization)",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        help="Resume from this stage (skip all stages before it)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print execution plan without running",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all available stages and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Pipeline entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # List stages and exit.
    if args.list_stages:
        stages = _build_stages()
        print("Available pipeline stages:")
        for i, stage in enumerate(stages, 1):
            print(f"  {i:2d}. {stage.name}")
        return 0

    # Resolve project root.
    project_root = resolve_project_root()

    # Load configuration.
    config = load_config(
        config_path=args.config,
        project_root=project_root,
    )

    # Validate configuration.
    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if not args.dry_run:
            return 1

    # Set up logging.
    log_dir = (
        Path(config.logging.log_dir)
        if Path(config.logging.log_dir).is_absolute()
        else project_root / config.logging.log_dir
    )
    setup_logging(
        level=config.logging.level,
        log_dir=log_dir if not args.dry_run else None,
        log_to_console=config.logging.log_to_console,
    )

    # Build context.
    ctx = PipelineContext(
        config=config,
        project_root=project_root,
        dry_run=args.dry_run,
    )

    # Build and run pipeline.
    stages = _build_stages()
    runner = PipelineRunner(stages)

    # Validate stage names if provided.
    if args.stage and args.stage not in runner.stage_names:
        print(
            f"Unknown stage: '{args.stage}'. "
            f"Available: {', '.join(runner.stage_names)}",
            file=sys.stderr,
        )
        return 1
    if args.from_stage and args.from_stage not in runner.stage_names:
        print(
            f"Unknown stage: '{args.from_stage}'. "
            f"Available: {', '.join(runner.stage_names)}",
            file=sys.stderr,
        )
        return 1

    results = runner.run_all(
        ctx,
        from_stage=args.from_stage,
        single_stage=args.stage,
    )

    # Return non-zero if any stage failed.
    failed = any(
        r.status == StageStatus.FAILED for r in results.values()
    )
    return 1 if failed else 0
