"""Reads whatever pipeline artifacts exist and shapes them for charting.

Primary source: ``<artifacts_dir>/report/pipeline_report.json``, written
by ``pipeline.stages.report.ReportStage`` -- it already aggregates every
stage's status/duration/metrics/artifacts, so this module mostly just
re-shapes that JSON for the frontend's charts.

Two things aren't in that aggregate report and are read directly:
- Per-epoch training curves, from Ultralytics' own ``results.csv``
  (written to ``<artifacts_dir>/training/<run_name>/results.csv``).
- Raw model file sizes (best.pt vs the quantized .tflite), read straight
  off disk via the artifact paths the report already recorded.

Every section is optional and independently tolerant of missing files --
a partially-completed or still-running pipeline should render whatever
is available rather than erroring out.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def artifacts_dir_from_config(config_path: str) -> Path:
    path = _resolve(config_path)
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return _resolve(config.get("artifacts_dir", "artifacts"))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _load_training_curve(artifacts_dir: Path, train_artifacts: dict[str, Any]) -> list[dict[str, float]]:
    train_dir = train_artifacts.get("train_dir")
    candidates = []
    if train_dir:
        candidates.append(Path(train_dir) / "results.csv")
    candidates.append(artifacts_dir / "training" / "train" / "results.csv")

    csv_path = next((c for c in candidates if c.is_file()), None)
    if csv_path is None:
        return []

    curve: list[dict[str, float]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row = {k.strip(): v for k, v in row.items()}

                def _f(*keys: str) -> float | None:
                    for k in keys:
                        if k in row and row[k] not in (None, ""):
                            try:
                                return float(row[k])
                            except ValueError:
                                continue
                    return None

                curve.append({
                    "epoch": _f("epoch"),
                    "box_loss": _f("train/box_loss"),
                    "cls_loss": _f("train/cls_loss"),
                    "dfl_loss": _f("train/dfl_loss"),
                    "precision": _f("metrics/precision(B)", "metrics/precision"),
                    "recall": _f("metrics/recall(B)", "metrics/recall"),
                    "map50": _f("metrics/mAP50(B)", "metrics/mAP50"),
                    "map50_95": _f("metrics/mAP50-95(B)", "metrics/mAP50-95"),
                })
    except OSError:
        return []
    return curve


def _model_sizes(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    sizes = []
    labels = {
        "best_weights": "Trained (.pt)",
        "onnx_model": "ONNX",
        "tflite_model": "Quantized (.tflite)",
        "nb_model": "Compiled (.nb)",
    }
    for key, label in labels.items():
        path_str = artifacts.get(key)
        if not path_str:
            continue
        path = Path(path_str)
        if path.is_file():
            sizes.append({
                "label": label,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            })
    return sizes


def load_results(config_path: str) -> dict[str, Any]:
    artifacts_dir = artifacts_dir_from_config(config_path)

    report = _load_json(artifacts_dir / "report" / "pipeline_report.json")
    stages = (report or {}).get("stages", {})

    stage_timeline = [
        {
            "name": name,
            "status": info.get("status"),
            "duration_seconds": info.get("duration_seconds") or 0,
            "message": info.get("message", ""),
        }
        for name, info in stages.items()
        if isinstance(info, dict) and "status" in info
    ]

    all_artifacts: dict[str, Any] = {}
    for info in stages.values():
        if isinstance(info, dict):
            all_artifacts.update(info.get("artifacts") or {})
    all_artifacts.update((report or {}).get("artifacts", {}))

    training_metrics = (stages.get("training") or {}).get("metrics") or {}
    if not training_metrics:
        training_metrics = _load_json(artifacts_dir / "training" / "training_metrics.json") or {}

    training_curve = _load_training_curve(
        artifacts_dir, (stages.get("training") or {}).get("artifacts") or {}
    )

    evaluation_metrics = (stages.get("evaluation") or {}).get("metrics") or {}
    if not evaluation_metrics:
        evaluation_metrics = _load_json(artifacts_dir / "evaluation" / "evaluation_report.json") or {}

    quantization_metrics = (stages.get("quantization") or {}).get("metrics") or {}

    benchmark_metrics = (stages.get("benchmark") or {}).get("metrics") or {}
    if not benchmark_metrics:
        benchmark_metrics = _load_json(artifacts_dir / "benchmark" / "benchmark_report.json") or {}

    return {
        "artifacts_dir": str(artifacts_dir),
        "has_report": report is not None,
        "stage_timeline": stage_timeline,
        "training_curve": training_curve,
        "training_metrics": training_metrics,
        "evaluation_metrics": evaluation_metrics,
        "quantization_metrics": quantization_metrics,
        "benchmark_metrics": benchmark_metrics,
        "model_sizes": _model_sizes(all_artifacts),
    }
