"""Stage 9: Deployment verification.

Verifies that the deployed model:
- Was uploaded successfully
- Loads correctly on the board
- Can run inference
- Uses the NPU (if applicable)

Verification is performed by running configured commands on the board
via SSH.  If no verification command is configured, the stage checks
that the model file exists on the remote filesystem.
"""

from __future__ import annotations

import logging
import subprocess

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class VerificationStage(Stage):
    """Verify deployment success on the STM32 board."""

    def __init__(self) -> None:
        super().__init__("verification")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if not ctx.config.deployment.enabled:
            return "Deployment is disabled; skipping verification"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.config.deployment.enabled:
            return errors

        cfg = ctx.config.deployment

        if not cfg.host:
            errors.append("deployment.host is required for verification")
        if not cfg.username:
            errors.append("deployment.username is required for verification")

        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        cfg = ctx.config.deployment
        checks_passed = 0
        checks_total = 0
        details: list[str] = []

        # Check 1: Model file exists on remote
        checks_total += 1
        model = self._find_deployed_model_name(ctx)
        if model:
            remote_path = f"{cfg.remote_model_directory}/{model}"
            exists = self._run_ssh(
                cfg, f"test -f {remote_path} && echo EXISTS"
            )
            if "EXISTS" in exists:
                details.append(f"Model file exists: {remote_path}")
                checks_passed += 1
            else:
                details.append(f"Model file NOT found: {remote_path}")
        else:
            details.append("No model name available to verify")

        # Check 2: Run verification command if configured
        if cfg.verification_command:
            checks_total += 1
            self.logger.info("Running verification: %s", cfg.verification_command)
            output = self._run_ssh(cfg, cfg.verification_command)
            if output is not None:
                details.append(f"Verification command output:\n{output}")
                checks_passed += 1
            else:
                details.append("Verification command failed")

        # Check 3: Run inference command if configured
        if cfg.inference_command:
            checks_total += 1
            self.logger.info("Running inference test: %s", cfg.inference_command)
            output = self._run_ssh(cfg, cfg.inference_command)
            if output is not None:
                details.append(f"Inference test output:\n{output}")
                checks_passed += 1
            else:
                details.append("Inference test failed")

        msg = f"Verification: {checks_passed}/{checks_total} checks passed"
        self.logger.info(msg)
        for detail in details:
            self.logger.info("  %s", detail)

        status = (
            StageStatus.SUCCESS
            if checks_passed == checks_total
            else StageStatus.FAILED
        )

        return StageResult(
            status=status,
            message=msg,
            metrics={
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "details": details,
            },
        )

    def _find_deployed_model_name(self, ctx: PipelineContext) -> str | None:
        """Get the filename of the deployed model."""
        for model in [ctx.nb_model, ctx.tflite_model, ctx.onnx_model]:
            if model is not None and model.is_file():
                return model.name
        return None

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
                timeout=300,
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
