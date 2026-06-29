"""Stage 2: Data augmentation.

Reuses the augmentation logic from ``scripts/augment_dataset.py``:
- Brightness adjustment (dark / bright)
- Contrast adjustment (low / high)
- Gaussian blur
- Gaussian noise

Geometric augmentations (rotation, perspective) are disabled by default
because the existing scripts do not transform labels alongside images.
When ``copy_labels`` is enabled, the corresponding label file is copied
unchanged for each augmented image (valid only for pixel-level transforms).
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..config.schema import AugmentationConfig
from ..core.context import PipelineContext
from ..core.stage import Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation functions (preserved from scripts/augment_dataset.py)
# ---------------------------------------------------------------------------

def change_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness by scaling the V channel in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def change_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """Adjust contrast via linear scaling."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def gaussian_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def add_noise(img: np.ndarray, stddev: float = 10.0) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, stddev, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotate_small(img: np.ndarray, max_degrees: float = 5.0) -> np.ndarray:
    """Small random rotation (WARNING: invalidates labels)."""
    h, w = img.shape[:2]
    angle = random.uniform(-max_degrees, max_degrees)
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def perspective_shift(img: np.ndarray, max_shift: int = 15) -> np.ndarray:
    """Random perspective warp (WARNING: invalidates labels)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.randint(0, max_shift), random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), random.randint(0, max_shift)],
        [random.randint(0, max_shift), h - random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), h - random.randint(0, max_shift)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class AugmentationStage(Stage):
    """Apply data augmentation to the training images."""

    def __init__(self) -> None:
        super().__init__("augmentation")

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if not ctx.config.augmentation.enabled:
            return "Augmentation is disabled in config"
        output_dir = ctx.stage_artifacts_dir(self.name)
        images_dir = output_dir / "images"
        if images_dir.is_dir() and any(images_dir.iterdir()):
            return f"Augmented images already exist in {images_dir}"
        return None

    def validate_inputs(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.config.augmentation.enabled:
            return errors

        cfg = ctx.config.dataset
        train_img_dir = ctx.dataset_dir / cfg.train_images
        if not train_img_dir.is_dir():
            # Fallback: flat images/ directory
            flat = ctx.dataset_dir / "images"
            if not flat.is_dir():
                errors.append(
                    f"Training images not found at {train_img_dir} or {flat}"
                )
        return errors

    def run(self, ctx: PipelineContext) -> StageResult:
        aug_cfg = ctx.config.augmentation
        ds_cfg = ctx.config.dataset
        output_dir = ctx.stage_artifacts_dir(self.name)
        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Resolve source directories
        train_img_dir = ctx.dataset_dir / ds_cfg.train_images
        train_lbl_dir = ctx.dataset_dir / ds_cfg.train_labels
        if not train_img_dir.is_dir():
            train_img_dir = ctx.dataset_dir / "images"
            train_lbl_dir = ctx.dataset_dir / "labels"

        ext_set = {e.lower() for e in ds_cfg.image_extensions}
        source_images = sorted(
            p for p in train_img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ext_set
        )

        if not source_images:
            return StageResult(
                status=StageStatus.FAILED,
                message=f"No images found in {train_img_dir}",
            )

        # Build augmentation list
        augmentations = self._build_augmentation_list(aug_cfg)
        self.logger.info(
            "Augmenting %d images with %d transforms",
            len(source_images),
            len(augmentations),
        )

        total_generated = 0
        for img_path in source_images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warning("Skipping unreadable image: %s", img_path)
                continue

            stem = img_path.stem

            # Copy original
            cv2.imwrite(str(images_out / f"{stem}_orig.jpg"), img)
            self._copy_label(stem, "_orig", train_lbl_dir, labels_out, aug_cfg)
            total_generated += 1

            # Apply each augmentation
            for aug_name, aug_fn in augmentations:
                augmented = aug_fn(img)
                out_name = f"{stem}_{aug_name}.jpg"
                cv2.imwrite(str(images_out / out_name), augmented)
                self._copy_label(stem, f"_{aug_name}", train_lbl_dir, labels_out, aug_cfg)
                total_generated += 1

        msg = (
            f"Generated {total_generated} images from {len(source_images)} "
            f"originals ({len(augmentations)} augmentations + original)"
        )
        self.logger.info(msg)

        return StageResult(
            status=StageStatus.SUCCESS,
            message=msg,
            artifacts={
                "images_dir": str(images_out),
                "labels_dir": str(labels_out),
            },
            metrics={
                "source_images": len(source_images),
                "total_generated": total_generated,
                "augmentations": [name for name, _ in augmentations],
            },
        )

    def _build_augmentation_list(
        self, cfg: AugmentationConfig
    ) -> list[tuple[str, callable]]:
        """Build the list of (name, function) augmentation pairs."""
        augmentations: list[tuple[str, callable]] = []

        for factor in cfg.brightness_factors:
            label = "dark" if factor < 1.0 else "bright"
            augmentations.append(
                (label, lambda img, f=factor: change_brightness(img, f))
            )

        for factor in cfg.contrast_factors:
            label = "contrast_low" if factor < 1.0 else "contrast_high"
            augmentations.append(
                (label, lambda img, f=factor: change_contrast(img, f))
            )

        if cfg.gaussian_blur:
            augmentations.append(
                ("blur", lambda img: gaussian_blur(img, cfg.gaussian_blur_kernel))
            )

        augmentations.append(
            ("noise", lambda img: add_noise(img, cfg.noise_stddev))
        )

        if cfg.rotate_enabled:
            self.logger.warning(
                "Rotation augmentation is enabled. Labels will NOT be "
                "transformed. This may produce incorrect training data."
            )
            augmentations.append(
                ("rotate", lambda img: rotate_small(img, cfg.rotate_max_degrees))
            )

        if cfg.perspective_enabled:
            self.logger.warning(
                "Perspective augmentation is enabled. Labels will NOT be "
                "transformed. This may produce incorrect training data."
            )
            augmentations.append(
                ("perspective", lambda img: perspective_shift(img, cfg.perspective_max_shift))
            )

        return augmentations

    def _copy_label(
        self,
        original_stem: str,
        suffix: str,
        source_labels_dir: Path,
        dest_labels_dir: Path,
        cfg: AugmentationConfig,
    ) -> None:
        """Copy the label file for an augmented image if it exists."""
        if not cfg.copy_labels:
            return
        src = source_labels_dir / f"{original_stem}.txt"
        if src.is_file():
            dst = dest_labels_dir / f"{original_stem}{suffix}.txt"
            shutil.copy2(src, dst)
