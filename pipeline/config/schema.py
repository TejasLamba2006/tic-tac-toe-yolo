"""Configuration dataclasses with validation.

Every pipeline setting lives here as a typed, documented field.
The ``PipelineConfig`` root class is built from a plain dictionary
(typically loaded from YAML) via ``PipelineConfig.from_dict()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Case-insensitive dict lookup with dotted-key support."""
    parts = key.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict):
            return default
        # Try exact key first, then lowercase match.
        if part in current:
            current = current[part]
        else:
            found = False
            for k, v in current.items():
                if k.lower() == part.lower():
                    current = v
                    found = True
                    break
            if not found:
                return default
    return current if current is not None else default


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Dataset paths and split behaviour."""

    path: str = "dataset4"
    train_images: str = "train/images"
    train_labels: str = "train/labels"
    val_images: str = "val/images"
    val_labels: str = "val/labels"
    test_images: str = "test/images"
    test_labels: str = "test/labels"
    image_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"],
    )
    split_enabled: bool = False
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1},
    )
    split_seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        defaults = cls()
        return cls(
            path=_get(data, "path", defaults.path),
            train_images=_get(data, "train_images", defaults.train_images),
            train_labels=_get(data, "train_labels", defaults.train_labels),
            val_images=_get(data, "val_images", defaults.val_images),
            val_labels=_get(data, "val_labels", defaults.val_labels),
            test_images=_get(data, "test_images", defaults.test_images),
            test_labels=_get(data, "test_labels", defaults.test_labels),
            image_extensions=_get(data, "image_extensions", defaults.image_extensions),
            split_enabled=_get(data, "split_enabled", defaults.split_enabled),
            split_ratios=_get(data, "split_ratios", defaults.split_ratios),
            split_seed=_get(data, "split_seed", defaults.split_seed),
        )


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""

    enabled: bool = True
    output_dir: str = "augmented"
    brightness_factors: list[float] = field(
        default_factory=lambda: [0.6, 1.4],
    )
    contrast_factors: list[float] = field(
        default_factory=lambda: [0.7, 1.4],
    )
    gaussian_blur: bool = True
    gaussian_blur_kernel: int = 5
    noise_stddev: float = 10.0
    # Geometric augmentations disabled by default because the existing
    # scripts do not transform labels alongside images.
    rotate_enabled: bool = False
    rotate_max_degrees: float = 5.0
    perspective_enabled: bool = False
    perspective_max_shift: int = 15
    copy_labels: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AugmentationConfig:
        defaults = cls()
        return cls(
            enabled=_get(data, "enabled", defaults.enabled),
            output_dir=_get(data, "output_dir", defaults.output_dir),
            brightness_factors=_get(data, "brightness_factors", defaults.brightness_factors),
            contrast_factors=_get(data, "contrast_factors", defaults.contrast_factors),
            gaussian_blur=_get(data, "gaussian_blur", defaults.gaussian_blur),
            gaussian_blur_kernel=_get(data, "gaussian_blur_kernel", defaults.gaussian_blur_kernel),
            noise_stddev=_get(data, "noise_stddev", defaults.noise_stddev),
            rotate_enabled=_get(data, "rotate_enabled", defaults.rotate_enabled),
            rotate_max_degrees=_get(data, "rotate_max_degrees", defaults.rotate_max_degrees),
            perspective_enabled=_get(data, "perspective_enabled", defaults.perspective_enabled),
            perspective_max_shift=_get(data, "perspective_max_shift", defaults.perspective_max_shift),
            copy_labels=_get(data, "copy_labels", defaults.copy_labels),
        )


@dataclass
class TrainingConfig:
    """YOLO training parameters."""

    model: str = "yolov8n.pt"
    data_yaml: str = "data.yaml"
    epochs: int = 100
    batch: int = 4
    imgsz: int = 320
    workers: int = 0
    patience: int = 100
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    device: str | None = None
    project: str | None = None
    name: str | None = None
    exist_ok: bool = True
    resume: bool = False
    seed: int = 0
    deterministic: bool = True
    amp: bool = True
    cache: bool = False
    plots: bool = True
    save: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        defaults = cls()
        return cls(**{
            f.name: _get(data, f.name, getattr(defaults, f.name))
            for f in cls.__dataclass_fields__.values()
        })


@dataclass
class ExportConfig:
    """ONNX export settings."""

    opset: int = 12
    imgsz: int = 320
    simplify: bool = False
    output_dir: str = "export"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportConfig:
        defaults = cls()
        return cls(**{
            f.name: _get(data, f.name, getattr(defaults, f.name))
            for f in cls.__dataclass_fields__.values()
        })


@dataclass
class QuantizationConfig:
    """TFLite INT8 quantization settings.

    Defaults match the existing ``scripts/config_quant.yaml``:
    per_channel, uint8 input, float output.
    """

    enabled: bool = True
    quantization_type: str = "per_channel"
    quantization_input_type: str = "uint8"
    quantization_output_type: str = "float"
    input_shape: list[int] = field(default_factory=lambda: [320, 320, 3])
    rescaling_scale: float = 255.0
    rescaling_offset: float = 0.0
    fake: bool = False
    calib_dataset_path: str = ""
    output_dir: str = "quantization"
    model_name: str = "tictactoe_yolov8"
    use_case: str = "od_tictactoe"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationConfig:
        defaults = cls()
        return cls(**{
            f.name: _get(data, f.name, getattr(defaults, f.name))
            for f in cls.__dataclass_fields__.values()
        })


@dataclass
class STEdgeConfig:
    """ST Edge AI Developer Cloud compilation settings."""

    enabled: bool = True
    target: str = ""
    optimization: str = ""
    board: str = ""
    runtime: str = ""
    output_dir: str = "compilation"
    # Credentials loaded from env vars or config.
    username: str = ""
    password: str = ""
    api_key: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> STEdgeConfig:
        defaults = cls()
        return cls(**{
            f.name: _get(data, f.name, getattr(defaults, f.name))
            for f in cls.__dataclass_fields__.values()
        })


@dataclass
class DeploymentConfig:
    """Remote board deployment settings."""

    enabled: bool = False
    host: str = ""
    port: int = 22
    username: str = ""
    password: str = ""
    ssh_key: str = ""
    remote_model_directory: str = ""
    remote_working_directory: str = ""
    inference_command: str = ""
    verification_command: str = ""
    benchmark_command: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentConfig:
        defaults = cls()
        return cls(**{
            f.name: _get(data, f.name, getattr(defaults, f.name))
            for f in cls.__dataclass_fields__.values()
        })


@dataclass
class BenchmarkConfig:
    """On-board benchmarking settings."""

    enabled: bool = False
    iterations: int = 100
    warmup_iterations: int = 10
    output_dir: str = "benchmark"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        defaults = cls()
        return cls(**{
            f.name: _get(data, f.name, getattr(defaults, f.name))
            for f in cls.__dataclass_fields__.values()
        })


@dataclass
class ReportConfig:
    """Report generation settings."""

    enabled: bool = True
    formats: list[str] = field(default_factory=lambda: ["json", "html", "markdown"])
    output_dir: str = "reports"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportConfig:
        defaults = cls()
        return cls(
            enabled=_get(data, "enabled", defaults.enabled),
            formats=_get(data, "formats", defaults.formats),
            output_dir=_get(data, "output_dir", defaults.output_dir),
        )


@dataclass
class LoggingConfig:
    """Logging behaviour."""

    level: str = "INFO"
    log_dir: str = "logs"
    log_to_console: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingConfig:
        defaults = cls()
        return cls(**{
            f.name: _get(data, f.name, getattr(defaults, f.name))
            for f in cls.__dataclass_fields__.values()
        })


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Root configuration for the entire pipeline."""

    artifacts_dir: str = "artifacts"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    stedge: STEdgeConfig = field(default_factory=STEdgeConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls(
            artifacts_dir=_get(data, "artifacts_dir", "artifacts"),
            dataset=DatasetConfig.from_dict(_get(data, "dataset", {})),
            augmentation=AugmentationConfig.from_dict(_get(data, "augmentation", {})),
            training=TrainingConfig.from_dict(_get(data, "training", {})),
            export=ExportConfig.from_dict(_get(data, "export", {})),
            quantization=QuantizationConfig.from_dict(_get(data, "quantization", {})),
            stedge=STEdgeConfig.from_dict(_get(data, "stedge", {})),
            deployment=DeploymentConfig.from_dict(_get(data, "deployment", {})),
            benchmark=BenchmarkConfig.from_dict(_get(data, "benchmark", {})),
            report=ReportConfig.from_dict(_get(data, "report", {})),
            logging=LoggingConfig.from_dict(_get(data, "logging", {})),
        )

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty means valid)."""
        errors: list[str] = []

        if not self.dataset.path:
            errors.append("dataset.path is required")

        if self.training.epochs < 1:
            errors.append("training.epochs must be >= 1")
        if self.training.batch < 1:
            errors.append("training.batch must be >= 1")
        if self.training.imgsz < 32:
            errors.append("training.imgsz must be >= 32")

        if self.export.opset < 7:
            errors.append("export.opset must be >= 7")

        ratios = self.dataset.split_ratios
        if self.dataset.split_enabled:
            total = sum(ratios.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"dataset.split_ratios must sum to 1.0 (got {total:.2f})"
                )

        if self.deployment.enabled:
            if not self.deployment.host:
                errors.append(
                    "deployment.host is required when deployment is enabled"
                )
            if not self.deployment.username:
                errors.append(
                    "deployment.username is required when deployment is enabled"
                )
            if not self.deployment.remote_model_directory:
                errors.append(
                    "deployment.remote_model_directory is required when "
                    "deployment is enabled"
                )

        return errors
