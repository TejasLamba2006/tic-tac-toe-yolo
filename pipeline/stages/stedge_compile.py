"""Stage 7: ST Edge AI Developer Cloud compilation.

Automates the generation of a Neural Binary (.nb) file from a quantized
TFLite model using the ST Edge AI Developer Cloud.

This stage uses the official ``stm32ai_dc`` SDK via our wrapper module
(``pipeline.stedge_wrapper``) for:
    - Authentication (SSO via my.st.com)
    - Model upload to the cloud
    - NBG generation for MPU targets (STM32MP1/STM32MP257)
    - Download of the optimized output

Credentials are loaded in this priority order:
    1. Environment variables (STEDGE_USERNAME, STEDGE_PASSWORD)
    2. Environment variables (STM32AI_USERNAME, STM32AI_PASSWORD)
    3. .env file
    4. config.yaml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class STEdgeCompileStage(Stage):
    """Compile a TFLite model to .nb using ST Edge AI Developer Cloud."""

    def __init__(self) -> None:
        super().__init__("stedge_compile")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if not ctx.config.stedge.enabled:
            return "ST Edge AI compilation is disabled in config"
        output_dir = ctx.stage_artifacts_dir(self.name)
        nb_files = list(output_dir.glob("*.nb"))
        if nb_files:
            ctx.nb_model = nb_files[0]
            return f"Neural binary already exists: {nb_files[0]}"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.config.stedge.enabled:
            return errors

        if ctx.tflite_model is None or not ctx.tflite_model.is_file():
            errors.append(
                "Quantized TFLite model not found. "
                "Run the quantization stage first."
            )

        # Check credentials (support both env var naming conventions)
        username = self._get_credential(ctx, "username", "STEDGE_USERNAME", "STM32AI_USERNAME")
        password = self._get_credential(ctx, "password", "STEDGE_PASSWORD", "STM32AI_PASSWORD")

        if not username:
            errors.append(
                "ST Edge AI credentials missing. Set STEDGE_USERNAME "
                "or STM32AI_USERNAME environment variable, add it to .env, "
                "or set stedge.username in config.yaml"
            )
        if not password:
            errors.append(
                "ST Edge AI credentials missing. Set STEDGE_PASSWORD "
                "or STM32AI_PASSWORD environment variable, add it to .env, "
                "or set stedge.password in config.yaml"
            )

        if not ctx.config.stedge.target:
            errors.append(
                "stedge.target is required (e.g., 'STM32MP257F-DK'). "
                "Set it in config.yaml."
            )

        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        cfg = ctx.config.stedge
        output_dir = ctx.stage_artifacts_dir(self.name)

        username = self._get_credential(ctx, "username", "STEDGE_USERNAME", "STM32AI_USERNAME")
        password = self._get_credential(ctx, "password", "STEDGE_PASSWORD", "STM32AI_PASSWORD")

        # Import wrapper (which imports stm32ai_dc)
        try:
            from ..stedge_wrapper import STEdgeAIClient
        except ImportError:
            return StageResult(
                status=StageStatus.FAILED,
                message=(
                    "The 'stm32ai_dc' package is not installed. "
                    "Install it with:\n"
                    "  pip install stm32ai_dc@git+https://github.com/STMicroelectronics/"
                    "stm32ai-modelzoo-services.git#subdirectory=common/stm32ai_dc\n"
                    "Or add to pyproject.toml:\n"
                    "  stm32ai_dc@git+https://github.com/STMicroelectronics/"
                    "stm32ai-modelzoo-services.git#subdirectory=common/stm32ai_dc"
                ),
            )

        self.logger.info(
            "Connecting to ST Edge AI Developer Cloud (target=%s)",
            cfg.target,
        )

        try:
            client = STEdgeAIClient(
                username=username,
                password=password,
            )
        except ValueError as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Authentication setup failed: {exc}",
            )
        except Exception as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=(
                    f"Failed to connect to ST Edge AI Developer Cloud: {exc}\n"
                    "Verify your credentials and network connection."
                ),
            )

        # Step 1: Upload model
        self.logger.info("Uploading %s...", ctx.tflite_model.name)
        try:
            upload_ok = client.upload_model(ctx.tflite_model)
            if not upload_ok:
                return StageResult(
                    status=StageStatus.FAILED,
                    message=f"Failed to upload {ctx.tflite_model.name} to the cloud",
                )
        except Exception as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Upload failed: {exc}",
            )

        # Step 2: Generate NBG (Neural Binary Graph) for MPU
        self.logger.info("Generating NBG for %s (timeout=600s)...", ctx.tflite_model.name)
        try:
            blob_name = client.generate_nbg(
                model_name=ctx.tflite_model.name,
                timeout=600,
            )
        except Exception as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=(
                    f"NBG generation failed: {exc}\n"
                    "Verify your target configuration and model compatibility."
                ),
            )

        # Step 3: Download the compiled .nb file
        self.logger.info("Downloading NBG: %s...", blob_name)
        try:
            nb_path = client.download_model(blob_name, output_dir)
            ctx.nb_model = nb_path

            msg = f"Neural binary compiled: {nb_path.name}"
            self.logger.info(msg)

            return StageResult(
                status=StageStatus.SUCCESS,
                message=msg,
                artifacts={"nb_model": str(nb_path)},
            )
        except Exception as exc:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"Download failed: {exc}",
            )

    def _get_credential(
        self, ctx: PipelineContext, config_key: str, env_key: str, fallback_env_key: str = ""
    ) -> str:
        """Resolve a credential from env vars, then config."""
        value = os.environ.get(env_key, "")
        if not value and fallback_env_key:
            value = os.environ.get(fallback_env_key, "")
        if value:
            return value
        return getattr(ctx.config.stedge, config_key, "")
