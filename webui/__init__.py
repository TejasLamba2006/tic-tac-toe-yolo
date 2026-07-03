"""Local web dashboard for the YOLOv8 -> STM32 pipeline.

This package is intentionally decoupled from ``pipeline/``: it never
imports pipeline internals directly, it only builds a YAML config file
in the exact shape ``pipeline.config.schema.PipelineConfig`` expects and
then runs ``run_pipeline.py --config <that file>`` as a subprocess,
exactly the way a user would from the CLI. That keeps the dashboard a
thin, low-risk layer on top of the existing, already-working pipeline.
"""
