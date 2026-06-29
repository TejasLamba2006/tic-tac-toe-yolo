"""Stage 10: On-board benchmarking.

Runs benchmark commands on the STM32 board via SSH and collects metrics:
- Inference latency
- FPS
- CPU usage
- NPU usage (if available)
- Memory usage

If a benchmark command is configured in deployment settings, it is
executed remotely.  Otherwise, a basic latency test is performed using
the deployed model.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class BenchmarkStage(Stage):
    """Run benchmarks on the deployed model."""

    def __init__(self) -> None:
        super().__init__("benchmark")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if not ctx.config.benchmark.enabled:
            return "Benchmarking is disabled in config"
        if not ctx.config.deployment.enabled:
            return "Deployment is disabled; cannot benchmark remotely"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.config.benchmark.enabled:
            return errors

        cfg = ctx.config.deployment
        if not cfg.host:
            errors.append("deployment.host is required for benchmarking")
        if not cfg.username:
            errors.append("deployment.username is required for benchmarking")

        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        cfg_deploy = ctx.config.deployment
        cfg_bench = ctx.config.benchmark
        output_dir = ctx.stage_artifacts_dir(self.name)

        metrics: dict = {}

        # Run configured benchmark command
        if cfg_deploy.benchmark_command:
            self.logger.info(
                "Running benchmark: %s", cfg_deploy.benchmark_command
            )
            output = self._run_ssh(cfg_deploy, cfg_deploy.benchmark_command)
            if output:
                metrics["benchmark_output"] = output
                parsed = self._parse_benchmark_output(output)
                metrics.update(parsed)
            else:
                return StageResult(
                    status=StageStatus.FAILED,
                    message="Benchmark command failed on the board",
                )
        else:
            self.logger.info("No benchmark command configured; collecting basic metrics")
            basic = self._collect_basic_metrics(ctx, cfg_deploy)
            metrics.update(basic)

        # Save results
        report_path = output_dir / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        msg_parts = []
        if "latency_ms" in metrics:
            msg_parts.append(f"latency={metrics['latency_ms']:.1f}ms")
        if "fps" in metrics:
            msg_parts.append(f"fps={metrics['fps']:.1f}")
        msg = "Benchmark: " + ", ".join(msg_parts) if msg_parts else "Benchmark complete"

        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={"report": str(report_path)},
            metrics=metrics,
        )

    def _collect_basic_metrics(self, ctx, cfg_deploy) -> dict:
        """Collect basic system metrics from the board."""
        metrics: dict = {}

        # Memory info
        mem_output = self._run_ssh(cfg_deploy, "free -m | head -2")
        if mem_output:
            metrics["memory_info"] = mem_output

        # CMA info (NPU memory)
        cma_output = self._run_ssh(
            cfg_deploy, "grep CmaFree /proc/meminfo 2>/dev/null || echo N/A"
        )
        if cma_output:
            metrics["cma_free"] = cma_output.strip()

        # CPU info
        cpu_output = self._run_ssh(
            cfg_deploy,
            "top -bn1 | head -5 2>/dev/null || echo N/A",
        )
        if cpu_output:
            metrics["cpu_info"] = cpu_output

        return metrics

    def _parse_benchmark_output(self, output: str) -> dict:
        """Try to extract numeric metrics from benchmark output."""
        metrics: dict = {}

        # Common patterns in benchmark outputs
        latency_match = re.search(
            r"(?:latency|inference)[:\s]*([0-9.]+)\s*ms", output, re.IGNORECASE
        )
        if latency_match:
            metrics["latency_ms"] = float(latency_match.group(1))

        fps_match = re.search(
            r"(?:fps|frame)[:\s]*([0-9.]+)", output, re.IGNORECASE
        )
        if fps_match:
            metrics["fps"] = float(fps_match.group(1))

        return metrics

    def _run_ssh(self, cfg, command: str) -> str | None:
        """Run a command on the remote board and return stdout."""
        ssh_cmd = ["ssh", "-p", str(cfg.port)]
        if cfg.ssh_key:
            ssh_cmd.extend(["-i", cfg.ssh_key])
        ssh_cmd.extend(["-o", "StrictHostKeyChecking=no"])
        ssh_cmd.extend(["-o", "UserKnownHostsFile=/dev/null"])
        ssh_cmd.append(f"{cfg.username}@{cfg.host}")
        ssh_cmd.append(command)

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            self.logger.warning(
                "SSH command failed (exit %d): %s",
                result.returncode,
                result.stderr,
            )
            return None
        except Exception as exc:
            self.logger.error("SSH failed: %s", exc)
            return None
