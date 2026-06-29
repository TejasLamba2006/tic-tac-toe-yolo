"""Stage 11: Report generation.

Generates pipeline summary reports in JSON, HTML, and Markdown formats.
Reads the pipeline manifest as its primary data source and aggregates
results from all previous stages into a single document.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class ReportStage(Stage):
    """Generate pipeline summary reports."""

    def __init__(self) -> None:
        super().__init__("report")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if not ctx.config.report.enabled:
            return "Report generation is disabled in config"
        return None

    def run(self, ctx: PipelineContext) -> StageResult:
        output_dir = ctx.stage_artifacts_dir(self.name)
        formats = ctx.config.report.formats

        report_data = self._build_report_data(ctx)
        generated: list[str] = []

        if "json" in formats:
            path = output_dir / "pipeline_report.json"
            self._write_json(report_data, path)
            generated.append(str(path))

        if "markdown" in formats:
            path = output_dir / "pipeline_report.md"
            self._write_markdown(report_data, path)
            generated.append(str(path))

        if "html" in formats:
            path = output_dir / "pipeline_report.html"
            self._write_html(report_data, path)
            generated.append(str(path))

        msg = f"Generated {len(generated)} report(s)"
        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={f"report_{i}": p for i, p in enumerate(generated)},
        )

    def _build_report_data(self, ctx: PipelineContext) -> dict[str, Any]:
        """Aggregate all stage results into a report dictionary."""
        # Prefer manifest data if available.
        manifest = getattr(ctx, "manifest", None)

        stages: dict[str, Any] = {}
        for name, result in ctx.stage_results.items():
            if hasattr(result, "status"):
                stages[name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_seconds": result.duration_seconds,
                    "artifacts": result.artifacts,
                    "metrics": self._serialize_metrics(result.metrics),
                }
            else:
                stages[name] = {"raw": str(result)}

        config_snapshot = {}
        if manifest:
            config_snapshot = manifest.config_snapshot
        else:
            config_snapshot = {
                "dataset_path": ctx.config.dataset.path,
                "model": ctx.config.training.model,
                "epochs": ctx.config.training.epochs,
                "imgsz": ctx.config.training.imgsz,
                "batch": ctx.config.training.batch,
                "quantization_type": ctx.config.quantization.quantization_type,
            }

        return {
            "pipeline_version": "0.1.0",
            "run_id": manifest.run_id if manifest else "unknown",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "project_root": str(ctx.project_root),
            "config": config_snapshot,
            "stages": stages,
            "artifacts": manifest.artifacts if manifest else {},
            "training_metrics": self._serialize_metrics(ctx.training_metrics),
            "metrics": manifest.metrics if manifest else {},
        }

    def _serialize_metrics(self, metrics: Any) -> Any:
        """Ensure metrics are JSON-serializable."""
        if isinstance(metrics, dict):
            return {
                k: self._serialize_metrics(v) for k, v in metrics.items()
            }
        if isinstance(metrics, (list, tuple)):
            return [self._serialize_metrics(v) for v in metrics]
        if isinstance(metrics, float):
            return round(metrics, 6)
        return metrics

    def _write_json(self, data: dict, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        self.logger.info("JSON report: %s", path)

    def _write_markdown(self, data: dict, path: Path) -> None:
        lines: list[str] = []
        lines.append("# Pipeline Report")
        lines.append("")
        lines.append(f"**Run ID:** {data.get('run_id', 'N/A')}")
        lines.append(f"**Generated:** {data['generated_at']}")
        lines.append(f"**Project:** {data['project_root']}")
        lines.append("")

        # Config summary
        lines.append("## Configuration")
        lines.append("")
        cfg = data.get("config", {})
        for key, value in cfg.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

        # Stage results
        lines.append("## Stage Results")
        lines.append("")
        lines.append("| Stage | Status | Duration | Message |")
        lines.append("|-------|--------|----------|---------|")
        for name, info in data.get("stages", {}).items():
            if isinstance(info, dict) and "status" in info:
                duration = f"{info['duration_seconds']:.1f}s"
                msg = info.get("message", "")[:60]
                lines.append(f"| {name} | {info['status']} | {duration} | {msg} |")
        lines.append("")

        # Artifacts
        artifacts = data.get("artifacts", {})
        if artifacts:
            lines.append("## Artifacts")
            lines.append("")
            for key, value in artifacts.items():
                lines.append(f"- **{key}:** `{value}`")
            lines.append("")

        # Training metrics
        training = data.get("training_metrics", {})
        if training:
            lines.append("## Training Metrics")
            lines.append("")
            for key, value in training.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info("Markdown report: %s", path)

    def _write_html(self, data: dict, path: Path) -> None:
        stages_rows = ""
        for name, info in data.get("stages", {}).items():
            if isinstance(info, dict) and "status" in info:
                status = info["status"]
                color_map = {
                    "success": "#22c55e",
                    "skipped": "#eab308",
                    "failed": "#ef4444",
                }
                color = color_map.get(status, "#6b7280")
                msg = info.get("message", "")[:80]
                stages_rows += (
                    f"<tr>"
                    f"<td>{name}</td>"
                    f'<td style="color:{color};font-weight:bold">{status}</td>'
                    f"<td>{info['duration_seconds']:.1f}s</td>"
                    f"<td>{msg}</td>"
                    f"</tr>\n"
                )

        training_rows = ""
        for key, value in data.get("training_metrics", {}).items():
            training_rows += f"<tr><td>{key}</td><td>{value}</td></tr>\n"

        artifacts_rows = ""
        for key, value in data.get("artifacts", {}).items():
            artifacts_rows += f"<tr><td>{key}</td><td><code>{value}</code></td></tr>\n"

        run_id = data.get("run_id", "N/A")
        generated = data.get("generated_at", "")
        project = data.get("project_root", "")

        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '    <meta charset="UTF-8">',
            "    <title>Pipeline Report</title>",
            "    <style>",
            "        body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f9fafb; }",
            "        h1 { color: #111827; }",
            "        h2 { color: #374151; margin-top: 32px; }",
            "        table { border-collapse: collapse; width: 100%; margin-top: 12px; }",
            "        th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }",
            "        th { background: #f3f4f6; font-weight: 600; }",
            "        .meta { color: #6b7280; font-size: 14px; }",
            "        code { background: #f3f4f6; padding: 2px 6px; border-radius: 3px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Pipeline Report</h1>",
            f'    <p class="meta">Run ID: {run_id}</p>',
            f'    <p class="meta">Generated: {generated}</p>',
            f'    <p class="meta">Project: {project}</p>',
            "",
            "    <h2>Stage Results</h2>",
            "    <table>",
            "        <tr><th>Stage</th><th>Status</th><th>Duration</th><th>Message</th></tr>",
            f"        {stages_rows}",
            "    </table>",
            "",
            "    <h2>Artifacts</h2>",
            "    <table>",
            "        <tr><th>Artifact</th><th>Path</th></tr>",
            f"        {artifacts_rows}",
            "    </table>",
            "",
            "    <h2>Training Metrics</h2>",
            "    <table>",
            "        <tr><th>Metric</th><th>Value</th></tr>",
            f"        {training_rows}",
            "    </table>",
            "</body>",
            "</html>",
        ]

        path.write_text("\n".join(html_parts), encoding="utf-8")
        self.logger.info("HTML report: %s", path)
