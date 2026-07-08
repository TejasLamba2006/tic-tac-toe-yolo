"""Dataset inspection, zip handling, and YAML config generation.

Produces a plain dict shaped exactly like ``pipeline/config/config.yaml``
(see ``pipeline.config.schema.PipelineConfig``) and writes it out with
``yaml.safe_dump``. This module never imports the ``pipeline`` package
so the dashboard has no import-time coupling to it -- the generated
YAML is the only contract between the two.
"""

from __future__ import annotations

import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = PROJECT_ROOT / "dataset_uploads"
CONFIGS_DIR = PROJECT_ROOT / "generated_configs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_CANDIDATES = {
    "train": ["train/images", "train"],
    "val": ["valid/images", "val/images", "valid", "val"],
    "test": ["test/images", "test"],
}
# Names of the standard split layout this module writes/reads (root/<split>/images).
# Excluded when re-scanning for source images so re-running split_dataset() is safe.
SPLIT_OUTPUT_DIRS = {"train", "val", "test"}


class DatasetError(ValueError):
    """Raised for invalid dataset uploads/paths."""


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("_")
    return slug or "dataset"


def _count_images(images_dir: Path) -> int:
    if not images_dir.is_dir():
        return 0
    return sum(
        1 for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_data_yaml(root: Path) -> Path | None:
    for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
        p = root / candidate
        if p.is_file():
            return p
    # One level deep too (common when a zip wraps everything in a folder).
    for p in root.glob("*/data.yaml"):
        return p
    return None


def extract_zip(zip_path: Path, dest_name: str) -> Path:
    """Safely extract *zip_path* into ``dataset_uploads/<slug>/``.

    Guards against zip-slip (entries that escape the destination
    directory via ``..`` or absolute paths).
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(dest_name)
    dest_dir = UPLOADS_DIR / slug
    suffix = 1
    while dest_dir.exists():
        suffix += 1
        dest_dir = UPLOADS_DIR / f"{slug}-{suffix}"
    dest_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = (dest_dir / member.filename).resolve()
            if not str(member_path).startswith(str(dest_dir.resolve())):
                raise DatasetError(
                    f"Rejected unsafe zip entry (path traversal): {member.filename}"
                )
        zf.extractall(dest_dir)

    # If everything landed inside a single wrapper folder, flatten it so
    # dataset.path points at the real root (common with Roboflow exports).
    children = [c for c in dest_dir.iterdir() if not c.name.startswith("__MACOSX")]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest_dir


def inspect_dataset(root: Path) -> dict[str, Any]:
    """Scan a dataset directory and report what the pipeline can use."""
    if not root.is_dir():
        raise DatasetError(f"Not a directory: {root}")

    splits: dict[str, dict[str, Any]] = {}
    for split, candidates in SPLIT_CANDIDATES.items():
        for rel in candidates:
            images_dir = root / rel
            count = _count_images(images_dir)
            if count > 0:
                splits[split] = {"images_dir": rel, "count": count}
                break

    data_yaml_path = _find_data_yaml(root)
    class_names: list[str] = []
    if data_yaml_path is not None:
        try:
            with open(data_yaml_path, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            names = loaded.get("names")
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                class_names = list(names)
        except (OSError, yaml.YAMLError):
            data_yaml_path = None

    total_images = sum(s["count"] for s in splits.values())
    if total_images == 0:
        # Fall back to a flat scan (no train/val/test structure yet).
        total_images = _count_images(root)

    return {
        "root": str(root),
        "splits": splits,
        "total_images": total_images,
        "has_split_layout": bool(splits),
        "data_yaml": str(data_yaml_path) if data_yaml_path else None,
        "class_names": class_names,
    }


def split_dataset(root: Path, ratios: dict[str, float], seed: int = 42) -> dict[str, int]:
    """Physically split a flat/non-standard-layout dataset into train/val/test.

    Recursively collects every image under *root* (skipping the standard
    ``train/``, ``val/``, ``test/`` output dirs so re-running is safe),
    matches each to a same-stem ``.txt`` label found anywhere under
    *root*, shuffles, and copies pairs into ``<split>/images`` +
    ``<split>/labels`` -- the layout ``inspect_dataset`` recognizes.
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 0.01:
        raise DatasetError(f"Split ratios must sum to 1.0 (got {total:.2f})")

    label_by_stem = {p.stem: p for p in root.rglob("*.txt")}

    images = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and p.relative_to(root).parts[0] not in SPLIT_OUTPUT_DIRS
    ]
    if not images:
        raise DatasetError(f"No images found under {root}")

    random.Random(seed).shuffle(images)

    n = len(images)
    n_train = int(n * ratios.get("train", 0.8))
    n_val = int(n * ratios.get("val", 0.1))
    buckets = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    counts: dict[str, int] = {}
    for split, files in buckets.items():
        if not files:
            continue
        img_dst = root / split / "images"
        lbl_dst = root / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, img_dst / img.name)
            label = label_by_stem.get(img.stem)
            if label is not None:
                shutil.copy2(label, lbl_dst / f"{img.stem}.txt")
        counts[split] = len(files)
    return counts


def generate_data_yaml(dataset_root: Path, class_names: list[str]) -> Path:
    """Write a minimal YOLO ``data.yaml`` when the dataset doesn't have one."""
    info = inspect_dataset(dataset_root)
    splits = info["splits"]

    def _dir_for(split: str, fallback: str) -> str:
        return splits.get(split, {}).get("images_dir", fallback)

    data = {
        "path": str(dataset_root),
        "train": _dir_for("train", "train/images"),
        "val": _dir_for("val", "valid/images"),
        "test": _dir_for("test", "test/images"),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    out_path = dataset_root / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out_path


def build_config_dict(form: dict[str, Any], dataset_info: dict[str, Any]) -> dict[str, Any]:
    """Map the dashboard form + dataset scan into a PipelineConfig-shaped dict."""
    name_slug = slugify(form["dataset_name"])
    dataset_root = Path(dataset_info["root"])
    imgsz = int(form["imgsz"])

    data_yaml_path = dataset_info.get("data_yaml")
    if not data_yaml_path:
        class_names = [
            c.strip() for c in form.get("class_names", "").split(",") if c.strip()
        ]
        if not class_names:
            raise DatasetError(
                "No data.yaml was found in the dataset and no class names were "
                "provided -- add comma-separated class names before generating "
                "the config."
            )
        data_yaml_path = str(generate_data_yaml(dataset_root, class_names))

    split_enabled = not dataset_info.get("has_split_layout", False)

    config: dict[str, Any] = {
        "artifacts_dir": f"artifacts_{name_slug}",
        "logging": {
            "level": "INFO",
            "log_dir": f"artifacts_{name_slug}/logs",
            "log_to_console": True,
        },
        "dataset": {
            "path": str(dataset_root),
            "train_images": "train/images",
            "train_labels": "train/labels",
            "val_images": "valid/images" if (dataset_root / "valid").exists() else "val/images",
            "val_labels": "valid/labels" if (dataset_root / "valid").exists() else "val/labels",
            "test_images": "test/images",
            "test_labels": "test/labels",
            "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
            "split_enabled": split_enabled,
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "split_seed": 42,
        },
        "augmentation": {"enabled": False},
        "training": {
            "model": "yolov8n.pt",
            "data_yaml": data_yaml_path,
            "epochs": int(form["epochs"]),
            "batch": int(form["batch_size"]),
            "imgsz": imgsz,
            "workers": int(form.get("workers", 0)),
            "patience": 100,
            "optimizer": "auto",
            # Default to CPU: forcing CUDA device 0 on a machine without a
            # GPU makes Ultralytics raise "Invalid CUDA 'device=0' requested"
            # before training even starts. The UI exposes an explicit
            # CPU/GPU choice; this is just a safe fallback if omitted.
            "device": form.get("device") or "cpu",
            "exist_ok": True,
            "resume": False,
            "seed": 0,
            "deterministic": True,
            "amp": True,
            "cache": False,
            "plots": True,
            "save": True,
        },
        "export": {
            "opset": 12,
            "imgsz": imgsz,
            "simplify": False,
            "output_dir": "export",
        },
        "quantization": {
            "enabled": bool(form.get("enable_int8", False)),
            "quantization_type": "per_channel",
            "quantization_input_type": "uint8",
            "quantization_output_type": "float",
            "input_shape": [imgsz, imgsz, 3],
            "rescaling_scale": 255.0,
            "rescaling_offset": 0.0,
            "fake": False,
            "calib_dataset_path": "",
            "output_dir": "quantization",
            "model_name": f"{name_slug}_yolov8",
            "use_case": f"od_{name_slug}",
        },
        "stedge": {
            "enabled": bool(form.get("deploy_stedge", False)),
            "target": form.get("stedge_target") or "STM32MP257F-DK",
            "optimization": "",
            "board": "",
            "runtime": "",
            "output_dir": "compilation",
            "username": form.get("stedge_username", "") if form.get("deploy_stedge") else "",
            "password": form.get("stedge_password", "") if form.get("deploy_stedge") else "",
            "timeout": 600,
        },
        "deployment": {"enabled": False},
        "benchmark": {"enabled": False},
        "report": {"enabled": True, "formats": ["json", "html", "markdown"], "output_dir": "reports"},
    }
    return config


def write_config_yaml(config: dict[str, Any], dataset_name: str) -> Path:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(dataset_name)
    out_path = CONFIGS_DIR / f"{slug}.yaml"
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return out_path


def redact_preview(config: dict[str, Any]) -> str:
    """YAML preview string with secrets masked for display in the browser."""
    import copy

    redacted = copy.deepcopy(config)
    stedge = redacted.get("stedge", {})
    if stedge.get("password"):
        stedge["password"] = "********"
    return yaml.safe_dump(redacted, sort_keys=False)
