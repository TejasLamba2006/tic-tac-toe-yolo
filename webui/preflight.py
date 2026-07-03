"""Preflight checks: everything that can be verified *before* spending
minutes/hours on a pipeline run that then dies on stage 3 with
``No module named 'ultralytics'``.

Two complementary sources of truth, deliberately not duplicated:
1. Package availability -- checked here via ``importlib.util.find_spec``
   (cheap: does not actually import heavy libs like torch/tensorflow).
2. Config/path validation -- delegated to the pipeline's own
   ``run_pipeline.py --config <path> --dry-run``, which already walks
   every stage's ``validate_inputs()`` (missing data.yaml, missing base
   model, bad split ratios, etc.). We don't reimplement that logic here.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (pip package name, import name, optional-dependency group or None if
# always required, human-readable purpose)
_PACKAGE_CHECKS: list[tuple[str, str, str | None, str]] = [
    ("pyyaml", "yaml", None, "Config loading (always required)"),
    ("opencv-python", "cv2", None, "Dataset validation / augmentation (always required)"),
    ("numpy", "numpy", None, "Numeric operations (always required)"),
    ("ultralytics", "ultralytics", "training", "YOLO training"),
    ("torch", "torch", "training", "YOLO training backend"),
    ("tensorflow", "tensorflow", "quantization", "INT8 quantization / TFLite export"),
    ("stm32ai_dc", "stm32ai_dc", "stedge", "ST Edge AI Developer Cloud compilation"),
]


def _check_package(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def check_packages(config: dict[str, Any]) -> list[dict[str, Any]]:
    quantization_enabled = bool((config.get("quantization") or {}).get("enabled"))
    stedge_enabled = bool((config.get("stedge") or {}).get("enabled"))

    checks = []
    for pip_name, import_name, group, label in _PACKAGE_CHECKS:
        if group == "quantization":
            required = quantization_enabled
        elif group == "stedge":
            required = stedge_enabled
        else:
            required = True
        available = _check_package(import_name)
        checks.append({
            "package": pip_name,
            "label": label,
            "required": required,
            "available": available,
            "ok": available or not required,
            "install_hint": f'pip install -e ".[{group}]"' if group else None,
        })
    return checks


def check_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = (config.get("dataset") or {}).get("path", "")
    data_yaml = (config.get("training") or {}).get("data_yaml", "")
    base_model = (config.get("training") or {}).get("model", "")

    dataset_dir = _resolve(dataset_path) if dataset_path else None
    data_yaml_path = _resolve(data_yaml) if data_yaml else None
    base_model_path = _resolve(base_model) if base_model else None

    return {
        "dataset_dir": str(dataset_dir) if dataset_dir else "",
        "dataset_dir_exists": bool(dataset_dir and dataset_dir.is_dir()),
        "data_yaml": str(data_yaml_path) if data_yaml_path else "",
        "data_yaml_exists": bool(data_yaml_path and data_yaml_path.is_file()),
        "base_model": str(base_model_path) if base_model_path else "",
        "base_model_exists": bool(base_model_path and base_model_path.is_file()),
    }


def check_disk(config: dict[str, Any]) -> dict[str, Any]:
    artifacts_dir = _resolve(config.get("artifacts_dir", "artifacts"))
    usage_target = artifacts_dir if artifacts_dir.exists() else PROJECT_ROOT
    total, _used, free = shutil.disk_usage(usage_target)
    free_gb = free / (1024 ** 3)
    return {
        "artifacts_dir": str(artifacts_dir),
        "free_gb": round(free_gb, 1),
        "total_gb": round(total / (1024 ** 3), 1),
        "low_space_warning": free_gb < 2.0,
    }


def _run_dry_run_blocking(config_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "run_pipeline.py", "--config", str(config_path), "--dry-run"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )


async def run_config_validation(config_path: Path) -> dict[str, Any]:
    """Run ``run_pipeline.py --config <path> --dry-run`` and report the result.

    This exercises the pipeline's real ``PipelineConfig.validate()`` plus
    every stage's ``validate_inputs()`` -- the same checks that would
    otherwise only surface minutes into a real run.

    Uses blocking ``subprocess.run`` inside a worker thread (via
    ``loop.run_in_executor``) rather than ``asyncio.create_subprocess_exec``:
    on Windows, uvicorn's reload / multi-worker supervisor forces
    ``SelectorEventLoop``, which does not support asyncio subprocess
    transports at all (raises ``NotImplementedError``). Plain
    ``subprocess.run`` has no such restriction.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, functools.partial(_run_dry_run_blocking, config_path)
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "output": result.stdout.decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "output": "Timed out after 30s while validating the config."}
    except OSError as exc:
        return {"ok": False, "returncode": None, "output": f"Failed to launch validator: {exc}"}


async def run_preflight(config_path: str) -> dict[str, Any]:
    path = _resolve(config_path)
    if not path.is_file():
        return {"error": f"Config file not found: {path}"}

    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    packages = check_packages(config)
    dataset = check_dataset(config)
    disk = check_disk(config)
    validation = await run_config_validation(path)

    overall_ok = (
        all(p["ok"] for p in packages)
        and not disk["low_space_warning"]
    )


    return {
        "config_path": str(path),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "packages": packages,
        "dataset": dataset,
        "disk": disk,
        "config_validation": validation,
        "overall_ok": overall_ok,
    }
