"""Stage 1: Dataset validation.

Checks for:
- Missing or empty label files
- Corrupt / unreadable images
- Unsupported image formats
- Invalid bounding boxes (out of range, malformed)
- Duplicate images (by filename stem)
- Class distribution statistics

Generates a validation report stored in the stage artifacts directory.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class DatasetValidationStage(Stage):
    """Validate the dataset before training."""

    def __init__(self) -> None:
        super().__init__("dataset_validation")

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.dataset_dir.exists():
            errors.append(f"Dataset directory not found: {ctx.dataset_dir}")
        return errors

    def should_skip(self, ctx: PipelineContext) -> str | None:
        report_path = ctx.stage_artifacts_dir(self.name) / "validation_report.json"
        if report_path.exists():
            return f"Validation report already exists: {report_path}"
        return None

    def run(self, ctx: PipelineContext) -> StageResult:
        cfg = ctx.config.dataset
        output_dir = ctx.stage_artifacts_dir(self.name)

        # Determine which image/label directories to validate.
        splits = self._discover_splits(ctx.dataset_dir, cfg)
        if not splits:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"No image directories found in {ctx.dataset_dir}",
            )

        report: dict[str, Any] = {
            "dataset_path": str(ctx.dataset_dir),
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "issues": [],
            "class_distribution": {},
        }

        all_class_counts: Counter[int] = Counter()
        all_issues: list[dict[str, str]] = []

        for split_name, images_dir, labels_dir in splits:
            self.logger.info("Validating split: %s", split_name)
            split_report = self._validate_split(
                split_name, images_dir, labels_dir, cfg.image_extensions
            )
            report["splits"][split_name] = split_report
            report["total_images"] += split_report["image_count"]
            report["total_labels"] += split_report["label_count"]
            all_issues.extend(split_report["issues"])
            for cls_id, count in split_report["class_counts"].items():
                all_class_counts[int(cls_id)] += count

        report["issues"] = all_issues
        report["class_distribution"] = dict(all_class_counts)

        # Write report
        report_path = output_dir / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        self.logger.info("Validation report: %s", report_path)

        # Summarize
        n_issues = len(all_issues)
        warnings = [i for i in all_issues if i["severity"] == "warning"]
        errors = [i for i in all_issues if i["severity"] == "error"]

        msg = (
            f"Validated {report['total_images']} images, "
            f"{report['total_labels']} labels. "
            f"{len(errors)} errors, {len(warnings)} warnings."
        )
        self.logger.info(msg)

        if errors:
            for e in errors[:10]:
                self.logger.error("  %s: %s", e["file"], e["message"])
            return StageResult(
                status=StageStatus.FAILED,
                message=msg,
                artifacts={"report": str(report_path)},
                metrics=report["class_distribution"],
            )

        if warnings:
            for w in warnings[:10]:
                self.logger.warning("  %s: %s", w["file"], w["message"])

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={"report": str(report_path)},
            metrics=report["class_distribution"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_splits(
        self,
        dataset_dir: Path,
        cfg: Any,
    ) -> list[tuple[str, Path, Path]]:
        """Find image/label directory pairs for each split."""
        splits: list[tuple[str, Path, Path]] = []

        for split_name, img_sub, lbl_sub in [
            ("train", cfg.train_images, cfg.train_labels),
            ("val", cfg.val_images, cfg.val_labels),
            ("test", cfg.test_images, cfg.test_labels),
        ]:
            img_dir = dataset_dir / img_sub
            lbl_dir = dataset_dir / lbl_sub
            if img_dir.is_dir():
                splits.append((split_name, img_dir, lbl_dir))

        # Fallback: flat dataset with images/ and labels/ at root.
        if not splits:
            flat_img = dataset_dir / "images"
            flat_lbl = dataset_dir / "labels"
            if flat_img.is_dir():
                splits.append(("all", flat_img, flat_lbl))

        return splits

    def _validate_split(
        self,
        split_name: str,
        images_dir: Path,
        labels_dir: Path,
        extensions: list[str],
    ) -> dict[str, Any]:
        ext_set = {e.lower() for e in extensions}
        images = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ext_set
        )

        issues: list[dict[str, str]] = []
        class_counts: Counter[int] = Counter()
        stems_seen: dict[str, Path] = {}
        label_count = 0

        for i, img_path in enumerate(images, start=1):
            if i % 1000 == 0 or i == len(images):
                self.logger.info("  %s: %d/%d images", split_name, i, len(images))

            # Duplicate check
            stem = img_path.stem
            if stem in stems_seen:
                issues.append({
                    "file": str(img_path),
                    "severity": "warning",
                    "message": f"Duplicate stem '{stem}' (also: {stems_seen[stem]})",
                })
            stems_seen[stem] = img_path

            # Readability check
            img = cv2.imread(str(img_path))
            if img is None:
                issues.append({
                    "file": str(img_path),
                    "severity": "error",
                    "message": "Image is corrupt or unreadable",
                })
                continue

            # Label check
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                issues.append({
                    "file": str(img_path),
                    "severity": "warning",
                    "message": "Missing label file",
                })
                continue

            label_count += 1
            label_text = label_path.read_text(encoding="utf-8").strip()
            if not label_text:
                issues.append({
                    "file": str(label_path),
                    "severity": "warning",
                    "message": "Label file is empty",
                })
                continue

            # Validate each bounding box line
            for line_num, line in enumerate(label_text.splitlines(), start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    issues.append({
                        "file": str(label_path),
                        "severity": "error",
                        "message": f"Line {line_num}: expected 5+ values, got {len(parts)}",
                    })
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                except ValueError:
                    issues.append({
                        "file": str(label_path),
                        "severity": "error",
                        "message": f"Line {line_num}: non-numeric values",
                    })
                    continue

                class_counts[cls_id] += 1

                # YOLO normalized coords should be in [0, 1]
                for val_name, val in [("cx", cx), ("cy", cy), ("w", bw), ("h", bh)]:
                    if val < 0.0 or val > 1.0:
                        issues.append({
                            "file": str(label_path),
                            "severity": "warning",
                            "message": (
                                f"Line {line_num}: {val_name}={val:.4f} "
                                f"outside [0, 1]"
                            ),
                        })

        return {
            "image_count": len(images),
            "label_count": label_count,
            "class_counts": dict(class_counts),
            "issues": issues,
        }
