"""Load and merge pipeline configuration from YAML, env vars, and .env files.

Priority (highest wins):
    1. Environment variables
    2. ``.env`` file
    3. ``config.yaml``
    4. Built-in defaults
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from .schema import PipelineConfig

logger = logging.getLogger(__name__)

# Env-var mapping: ENV_NAME -> dotted config key
_ENV_MAP: dict[str, str] = {
    "STEDGE_USERNAME": "stedge.username",
    "STEDGE_PASSWORD": "stedge.password",
    "STEDGE_API_KEY": "stedge.api_key",
    "PIPELINE_LOG_LEVEL": "logging.level",
}


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple ``.env`` file (KEY=VALUE lines, no shell expansion)."""
    env_vars: dict[str, str] = {}
    if not path.is_file():
        return env_vars
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            env_vars[key] = value
    return env_vars


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key like ``stedge.username``."""
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _apply_env_overrides(
    data: dict[str, Any],
    dotenv_vars: dict[str, str],
) -> None:
    """Apply environment variable overrides into the raw config dict."""
    for env_name, config_key in _ENV_MAP.items():
        # Real env vars take precedence over .env file values.
        value = os.environ.get(env_name) or dotenv_vars.get(env_name)
        if value:
            _set_nested(data, config_key, value)
            logger.debug("Config override from env: %s -> %s", env_name, config_key)


def load_config(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> PipelineConfig:
    """Load the pipeline configuration.

    Parameters
    ----------
    config_path:
        Explicit path to a YAML config file.  If ``None``, looks for
        ``pipeline/config/config.yaml`` relative to *project_root*.
    project_root:
        Project root directory.  Defaults to the current working directory.
    """
    if project_root is None:
        project_root = Path.cwd()

    # 1. Load YAML
    raw: dict[str, Any] = {}
    if config_path is not None:
        yaml_path = Path(config_path)
    else:
        yaml_path = project_root / "pipeline" / "config" / "config.yaml"

    if yaml_path.is_file():
        logger.info("Loading config from %s", yaml_path)
        with open(yaml_path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                raw = loaded
    else:
        logger.warning(
            "Config file not found at %s; using defaults", yaml_path
        )

    # 2. Load .env
    dotenv_path = project_root / ".env"
    dotenv_vars = _load_dotenv(dotenv_path)
    if dotenv_vars:
        logger.info("Loaded %d vars from %s", len(dotenv_vars), dotenv_path)

    # 3. Apply env overrides
    _apply_env_overrides(raw, dotenv_vars)

    # 4. Build typed config
    config = PipelineConfig.from_dict(raw)
    return config
