"""Typed configuration for training experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class DataConfig:
    """Input data used by the experiment."""

    train_path: str = "data/generated/h1/seed_30072026/train.jsonl"
    test_path: str = "data/generated/h1/seed_30072026/test.jsonl"

    def __post_init__(self) -> None:
        if not self.train_path:
            raise ValueError("data.train_path must not be empty")
        if not self.test_path:
            raise ValueError("data.test_path must not be empty")


@dataclass(slots=True)
class ModelConfig:
    """Model and tokenizer source."""

    name_or_path: str = ".models/Qwen3-0.6B"
    revision: str | None = None
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not self.name_or_path:
            raise ValueError("model.name_or_path must not be empty")


@dataclass(slots=True)
class TrainingConfig:
    """Parameters that affect optimization."""

    batch_size: int = 16
    learning_rate: float = 2e-5
    gradient_accumulation_steps: int = 1
    smoke_steps: int | None = 10
    epochs: int = 1
    seed: int = 42
    precision: str = "fp32"
    max_sequence_length: int | None = None
    lr_scheduler: str = "constant"
    final_learning_rate_ratio: float = 1.0
    warmup_ratio: float = 0.0
    warmup_min_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.smoke_steps is not None and self.smoke_steps <= 0:
            raise ValueError("training.smoke_steps must be positive or null")
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("training.precision must be one of: fp32, bf16")
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("training.max_sequence_length must be positive or null")
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError("training.lr_scheduler must be one of: constant, cosine")
        if not 0 < self.final_learning_rate_ratio <= 1:
            raise ValueError(
                "training.final_learning_rate_ratio must be greater than 0 and at most 1"
            )
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("training.warmup_ratio must be at least 0 and less than 1")
        if self.warmup_min_steps < 0:
            raise ValueError("training.warmup_min_steps must be non-negative")
        if not 0 < self.adam_beta1 < 1:
            raise ValueError("training.adam_beta1 must be between 0 and 1")
        if not 0 < self.adam_beta2 < 1:
            raise ValueError("training.adam_beta2 must be between 0 and 1")
        if self.adam_epsilon <= 0:
            raise ValueError("training.adam_epsilon must be positive")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm must be positive or null")


@dataclass(slots=True)
class EvaluationConfig:
    """Parameters for generated-answer evaluation."""

    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("evaluation.batch_size must be positive")


@dataclass(slots=True)
class RuntimeConfig:
    """Machine-specific and diagnostic settings."""

    device: str = "auto"
    debug: bool = False
    prediction_log_limit: int | None = None
    output_dir: str | None = None
    save_model: bool = True

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("runtime.device must not be empty")
        if self.prediction_log_limit is not None and self.prediction_log_limit <= 0:
            raise ValueError("runtime.prediction_log_limit must be positive or null")
        if self.output_dir is not None and not self.output_dir:
            raise ValueError("runtime.output_dir must be non-empty or null")


@dataclass(slots=True)
class TrackingConfig:
    """Optional ClearML experiment tracking."""

    enabled: bool = False
    project_name: str = "do-models-only-memorize"
    project_id: str | None = None
    task_name: str | None = None
    output_uri: str | bool | None = False
    log_interval_steps: int = 1
    tags: list[str] = field(default_factory=lambda: ["sft", "smoke"])

    def __post_init__(self) -> None:
        if self.enabled and not self.project_name and not self.project_id:
            raise ValueError(
                "tracking.project_name or tracking.project_id must be set when tracking is enabled"
            )
        if self.log_interval_steps <= 0:
            raise ValueError("tracking.log_interval_steps must be positive")


@dataclass(slots=True)
class ExperimentConfig:
    """Complete, serializable description of one experiment run."""

    experiment_name: str = "h1-sft-smoke"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for trackers and metadata files."""

        return asdict(self)


def config_from_mapping(values: Mapping[str, Any] | DictConfig) -> ExperimentConfig:
    """Validate a mapping against the structured schema and build dataclasses."""

    schema = OmegaConf.structured(ExperimentConfig)
    merged = OmegaConf.merge(schema, values)
    OmegaConf.resolve(merged)
    result = OmegaConf.to_object(merged)
    if not isinstance(result, ExperimentConfig):  # pragma: no cover - defensive guard
        raise TypeError("configuration did not resolve to ExperimentConfig")
    return result


def load_experiment_config(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Load YAML and apply optional ``section.key=value`` overrides."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config does not exist: {config_path}")

    file_config = OmegaConf.load(config_path)
    if not isinstance(file_config, DictConfig):
        raise TypeError(f"experiment config must contain a YAML mapping: {config_path}")
    override_config = OmegaConf.from_dotlist(list(overrides))
    merged = OmegaConf.merge(file_config, override_config)
    if not isinstance(merged, DictConfig):  # pragma: no cover - guarded by inputs above
        raise TypeError("merged experiment config is not a mapping")
    return config_from_mapping(merged)
