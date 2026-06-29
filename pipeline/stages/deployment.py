"""Stage 8: Deploy model to STM32 board via SSH/SCP.

All deployment parameters are loaded from configuration.
No values are hardcoded.  If required fields are missing, the stage
fails with a clear error message explaining what to configure.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class DeploymentStage(Stage):
    """Deploy the compiled model to the STM32 board."""

    def __init__(self) -> None:
        super().__init__("deployment")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if not ctx.config.deployment.enabled:
            return "Deployment is disabled in config"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.config.deployment.enabled:
            return errors

        cfg = ctx.config.deployment

        if not cfg.host:
            errors.append(
                "deployment.host is required. Set the board IP address "
                "in config.yaml."
            )
        if not cfg.username:
            errors.append(
                "deployment.username is required. Set the SSH username "
                "in config.yaml."
            )
        if not cfg.remote_model_directory:
            errors.append(
                "deployment.remote_model_directory is required. Set the "
                "target directory on the board in config.yaml."
            )

        # Need at least one model to deploy
        model = self._find_model_to_deploy(ctx)
        if model is None:
            errors.append(
                "No model artifact found to deploy. Run export, "
                "quantization, or compilation first."
            )

        # Check SSH tools are available
        if not self._ssh_available():
            errors.append(
                "SSH client (ssh/scp) not found on PATH. "
                "Install OpenSSH or ensure it is in your PATH."
            )

        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        cfg = ctx.config.deployment
        output_dir = ctx.stage_artifacts_dir(self.name)

        model_path = self._find_model_to_deploy(ctx)
        if model_path is None:
            return StageResult(
                status=StageStatus.FAILED,
                message="No model artifact found to deploy",
            )

        self.logger.info(
            "Deploying %s to %s@%s:%s",
            model_path.name,
            cfg.username,
            cfg.host,
            cfg.remote_model_directory,
        )

        # Build SCP command
        scp_target = f"{cfg.username}@{cfg.host}:{cfg.remote_model_directory}/"
        scp_cmd = self._build_scp_command(cfg, model_path, scp_target)

        self.logger.info("Running: %s", " ".join(scp_cmd))
        try:
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    message=f"SCP failed (exit {result.returncode}): {result.stderr}",
                )
        except subprocess.TimeoutExpired:
            return StageResult(
                status=StageStatus.FAILED,
                message="SCP timed out after 120 seconds",
            )
        except FileNotFoundError:
            return StageResult(
                status=StageStatus.FAILED,
                message="SCP command not found. Install OpenSSH.",
            )

        self.logger.info("Upload complete")

        # Run verification command if configured
        if cfg.verification_command:
            self.logger.info("Running verification command...")
            verify_result = self._run_remote_command(
                cfg, cfg.verification_command
            )
            if not verify_result:
                return StageResult(
                    status=StageStatus.FAILED,
                    message="Verification command failed on the board",
                )

        msg = f"Deployed {model_path.name} to {cfg.host}:{cfg.remote_model_directory}"
        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={"deployed_model": str(model_path)},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_model_to_deploy(self, ctx: PipelineContext) -> Path | None:
        """Find the best model artifact to deploy (prefer .nb > .tflite > .onnx)."""
        for model in [ctx.nb_model, ctx.tflite_model, ctx.onnx_model]:
            if model is not None and model.is_file():
                return model
        return None

    def _build_scp_command(
        self, cfg, local_path: Path, remote_target: str
    ) -> list[str]:
        """Build the SCP command with appropriate auth options."""
        cmd = ["scp", "-P", str(cfg.port)]

        if cfg.ssh_key:
            cmd.extend(["-i", cfg.ssh_key])

        # Disable strict host key checking for automated pipelines
        cmd.extend(["-o", "StrictHostKeyChecking=no"])
        cmd.extend(["-o", "UserKnownHostsFile=/dev/null"])

        cmd.append(str(local_path))
        cmd.append(remote_target)
        return cmd

    def _run_remote_command(self, cfg, command: str) -> bool:
        """Execute a command on the remote board via SSH."""
        ssh_cmd = ["ssh", "-p", str(cfg.port)]

        if cfg.ssh_key:
            ssh_cmd.extend(["-i", cfg.ssh_key])

        ssh_cmd.extend(["-o", "StrictHostKeyChecking=no"])
        ssh_cmd.extend(["-o", "UserKnownHostsFile=/dev/null"])
        ssh_cmd.append(f"{cfg.username}@{cfg.host}")
        ssh_cmd.append(command)

        self.logger.info("SSH: %s", " ".join(ssh_cmd))
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.stdout:
                self.logger.info("Remote stdout:\n%s", result.stdout)
            if result.stderr:
                self.logger.warning("Remote stderr:\n%s", result.stderr)
            return result.returncode == 0
        except Exception as exc:
            self.logger.error("SSH command failed: %s", exc)
            return False

    def _ssh_available(self) -> bool:
        """Check if SSH/SCP tools are available."""
        import shutil
        return shutil.which("scp") is not None
