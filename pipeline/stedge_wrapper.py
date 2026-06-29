"""Wrapper around the official stm32ai_dc SDK.

Provides a clean abstraction for our pipeline's workflow:
    Upload → Analyze → Generate NBG → Download

Handles authentication, retries, and structured logging.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_credentials(
    username: str | None = None,
    password: str | None = None,
) -> tuple[str, str]:
    """Resolve credentials from args, env vars, or stored tokens.

    Priority:
        1. Explicit arguments
        2. STEDGE_USERNAME / STEDGE_PASSWORD env vars (legacy)
        3. STM32AI_USERNAME / STM32AI_PASSWORD env vars (official SDK)
    """
    if not username:
        username = os.environ.get("STEDGE_USERNAME") or os.environ.get("STM32AI_USERNAME", "")
    if not password:
        password = os.environ.get("STEDGE_PASSWORD") or os.environ.get("STM32AI_PASSWORD", "")
    return username, password


def _create_backend(
    username: str,
    password: str,
    platform: Any = None,
    version: str | None = None,
) -> Any:
    """Create a CloudBackend with proper error handling."""
    try:
        from common.stm32ai_dc import CloudBackend
        from common.stm32ai_dc.types import BackendVersionType
    except ImportError:
        from stm32ai_dc import CloudBackend
        from stm32ai_dc.types import BackendVersionType

    if platform is None:
        platform = BackendVersionType.STM32MPU

    return CloudBackend(
        username=username,
        password=password,
        version=version,
        platform=platform,
    )


def _retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 2.0,
    description: str = "operation",
):
    """Execute a function with exponential backoff retry logic."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    description, attempt + 1, max_retries, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    description, max_retries, exc,
                )
    raise last_exc


class STEdgeAIClient:
    """High-level client wrapping the official stm32ai_dc SDK.

    Usage::

        client = STEdgeAIClient(username="user", password="pass")
        client.upload_model("model.tflite")
        blob_name = client.generate_nbg("model.tflite", timeout=600)
        client.download_model(blob_name, "./output/")
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        version: str | None = None,
        platform: Any = None,
    ) -> None:
        username, password = _resolve_credentials(username, password)

        if not username or not password:
            raise ValueError(
                "ST Edge AI credentials required. Set STEDGE_USERNAME/STEDGE_PASSWORD "
                "or STM32AI_USERNAME/STM32AI_PASSWORD environment variables."
            )

        self._username = username
        self._backend = _retry_with_backoff(
            lambda: _create_backend(username, password, platform, version),
            max_retries=3,
            base_delay=2.0,
            description="Authentication",
        )

        try:
            from common.stm32ai_dc import Stm32Ai
        except ImportError:
            from stm32ai_dc import Stm32Ai
        self._ai = Stm32Ai(self._backend)
        logger.info("Connected to ST Edge AI Developer Cloud")

    def upload_model(self, model_path: str | Path) -> bool:
        """Upload a model file to the cloud.

        Parameters
        ----------
        model_path:
            Local path to the model file (.tflite, .onnx, .h5).

        Returns
        -------
        bool
            True if upload succeeded.
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Uploading model: %s", model_path.name)
        result = _retry_with_backoff(
            lambda: self._ai.upload_model(str(model_path)),
            max_retries=3,
            base_delay=2.0,
            description=f"Upload {model_path.name}",
        )
        if result:
            logger.info("Upload complete: %s", model_path.name)
        else:
            logger.error("Upload failed: %s", model_path.name)
        return result

    def analyze(self, model_name: str, model_type: str = "tflite", **kwargs) -> Any:
        """Analyze a model for memory footprint and complexity.

        Parameters
        ----------
        model_name:
            Filename of the already-uploaded model.
        model_type:
            Model type string: "tflite", "onnx", or "keras".
        **kwargs:
            Additional CliParameters fields.
        """
        try:
            from common.stm32ai_dc import CliParameters, CliParameterType
        except ImportError:
            from stm32ai_dc import CliParameters, CliParameterType

        type_map = {
            "tflite": CliParameterType.TFLITE,
            "onnx": CliParameterType.ONNX,
            "keras": CliParameterType.KERAS,
        }
        cli_type = type_map.get(model_type)

        options = CliParameters(
            model=model_name,
            type=cli_type,
            **kwargs,
        )

        logger.info("Analyzing model: %s", model_name)
        result = _retry_with_backoff(
            lambda: self._ai.analyze(options),
            max_retries=2,
            base_delay=3.0,
            description=f"Analyze {model_name}",
        )
        logger.info("Analysis complete: ROM=%d bytes, RAM=%d bytes, MACC=%d",
                     result.rom_size, result.ram_size, result.macc)
        return result

    def generate_nbg(self, model_name: str, timeout: int = 300) -> str:
        """Generate a Neural Binary Graph for MPU targets.

        The model must already be uploaded via ``upload_model()``.

        Parameters
        ----------
        model_name:
            Filename of the already-uploaded model.
        timeout:
            Maximum seconds to wait for NBG generation.

        Returns
        -------
        str
            The blob name of the generated NBG file.
        """
        logger.info("Generating NBG for: %s (timeout=%ds)", model_name, timeout)
        blob_name = _retry_with_backoff(
            lambda: self._ai.generate_nbg(model_name, timeout=timeout),
            max_retries=2,
            base_delay=5.0,
            description=f"NBG generation for {model_name}",
        )
        logger.info("NBG generated: %s", blob_name)
        return blob_name

    def download_model(self, model_name: str, target_dir: str | Path) -> Path:
        """Download a model from the cloud to a local directory.

        Parameters
        ----------
        model_name:
            The cloud model name (e.g., blob name from generate_nbg).
        target_dir:
            Local directory to save the model.

        Returns
        -------
        Path
            Path to the downloaded file.
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / model_name

        logger.info("Downloading: %s → %s", model_name, target_file)
        _retry_with_backoff(
            lambda: self._ai.download_model(model_name, str(target_file)),
            max_retries=3,
            base_delay=2.0,
            description=f"Download {model_name}",
        )
        downloaded = target_file
        if downloaded.exists():
            logger.info("Download complete: %s", downloaded)
            return downloaded

        # Fallback: find any file in target_dir
        files = list(target_dir.iterdir())
        if files:
            return files[0]
        raise FileNotFoundError(f"Downloaded model not found in {target_dir}")

    def benchmark(
        self,
        model_name: str,
        board_name: str,
        timeout: int = 600,
        engine: str = "cpu",
        nb_cores: int = 1,
    ) -> Any:
        """Benchmark a model on a specific board.

        Parameters
        ----------
        model_name:
            Filename of the already-uploaded model.
        board_name:
            Target board name (from get_benchmark_boards()).
        timeout:
            Maximum seconds to wait.
        engine:
            "cpu" or "hw_accelerator".
        nb_cores:
            Number of cores for multi-core MPU.

        Returns
        -------
        BenchmarkResult or MpuBenchmarkResult
        """
        try:
            from common.stm32ai_dc import MpuParameters, MpuEngine
        except ImportError:
            from stm32ai_dc import MpuParameters, MpuEngine

        engine_map = {
            "cpu": MpuEngine.CPU,
            "hw_accelerator": MpuEngine.HW_ACCELERATOR,
        }
        options = MpuParameters(
            model=model_name,
            engine=engine_map.get(engine, MpuEngine.CPU),
            nbCores=nb_cores,
        )

        logger.info("Benchmarking %s on %s", model_name, board_name)
        result = _retry_with_backoff(
            lambda: self._ai.benchmark(options, board_name, timeout=timeout),
            max_retries=2,
            base_delay=5.0,
            description=f"Benchmark {model_name}",
        )
        logger.info("Benchmark complete: duration=%dms", result.duration_ms)
        return result

    def get_boards(self) -> list:
        """List available benchmark boards."""
        return self._ai.get_benchmark_boards()

    def list_models(self) -> list:
        """List models uploaded to the cloud."""
        return self._ai.list_models()

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from the cloud."""
        return self._ai.delete_model(model_name)
