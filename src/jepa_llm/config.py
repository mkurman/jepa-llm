"""Configuration schema and loader for JEPA fine-tuning runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class GeneralConfig:
    """General runtime options that apply across the training pipeline."""

    debug: int = 0
    """Verbosity level used by dataset preparation and trainers."""

    finetune_seed: int = 42
    """Random seed applied to dataset shuffling and trainer setup."""


@dataclass
class ModelConfig:
    """Model specific options such as LoRA usage and trust settings."""

    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    """Model identifier or path to load from the Hugging Face hub."""

    use_lora: bool = False
    """Whether to enable LoRA adapters instead of full fine-tuning."""

    lora_rank: int = 16
    """Rank used when initialising LoRA adapters."""

    pretrain: bool = False
    """Start from an initial configuration instead of loading pre-trained weights."""

    cache_dir: Optional[str] = None
    """Optional directory where model artefacts should be cached."""

    trust_remote_code: bool = True
    """Forward the trust flag when loading custom model implementations."""


@dataclass
class DatasetConfig:
    """Input data definition and preprocessing flags."""

    train_file: Optional[str] = None
    """Path to the training split; mutually exclusive with ``data_file``."""

    eval_file: Optional[str] = None
    """Optional evaluation split path when ``train_file`` is supplied."""

    data_file: Optional[str] = None
    """Single dataset file to be split according to ``eval_split``."""

    eval_split: float = 0.2
    """Fraction reserved for evaluation when ``data_file`` is provided."""

    split_seed: int = 42
    """Seed applied to dataset splitting operations."""

    max_items: Optional[int] = None
    """Limit the number of samples drawn from the dataset (useful for debugging)."""

    max_length: int = 512
    """Maximum sequence length for tokenisation."""

    predictors: int = 0
    """Number of predictor tokens appended to user prompts."""

    train_all: bool = False
    """Whether to compute loss across all tokens rather than assistant only."""

    plain: bool = False
    """Skip chat template application when formatting examples."""

    remove_thinking: bool = True
    """Strip hidden reasoning tags such as ``<think>`` from inputs."""

    cache_dir: Optional[str] = None
    """Cache directory passed to ``datasets.load_dataset``."""


@dataclass
class TrainingConfig:
    """Fine-tuning hyperparameters and trainer behaviour toggles."""

    output_dir: str = "./llama3-1b-fted"
    """Target directory where checkpoints and the final model are written."""

    batch_size: int = 4
    """Per device batch size."""

    grad_accum: int = 4
    """Number of gradient accumulation steps."""

    learning_rate: float = 2e-5
    """Optimiser learning rate."""

    num_epochs: int = 3
    """Number of full training epochs."""

    eval_steps: int = 10
    """Frequency (in steps) of evaluation and checkpointing."""

    lbd: float = 0.1
    """Weight applied to the JEPA similarity loss component."""

    gamma: float = 1.0
    """Weight applied to the language modelling loss component."""

    last_token: int = -1
    """Relative index used for embedding extraction in the JEPA loss."""

    regular: bool = False
    """Fallback to the vanilla Hugging Face trainer instead of JEPA trainer."""

    track_flop: bool = False
    """Enable FLOP profiling callback during the first few steps."""

    additive_mask: bool = False
    """Use additive masking to combine user and assistant sequences."""

    memory_efficient: bool = False
    """Process sequences separately to lower VRAM usage."""


@dataclass
class Config:
    """Aggregate configuration loaded from disk."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _build_dataclass(cls: Type[T], values: Optional[Dict[str, Any]]) -> T:
    values = values or {}
    allowed_keys = {field_.name for field_ in fields(cls)}
    unknown = set(values) - allowed_keys
    if unknown:
        raise ValueError(
            f"Unknown configuration options for {cls.__name__}: {', '.join(sorted(unknown))}"
        )
    return cls(**{key: values[key] for key in allowed_keys if key in values})


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file into typed dataclasses."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    if not isinstance(raw_data, dict):
        raise ValueError("Configuration root must be a mapping of sections")

    return Config(
        general=_build_dataclass(GeneralConfig, raw_data.get("general")),
        model=_build_dataclass(ModelConfig, raw_data.get("model")),
        dataset=_build_dataclass(DatasetConfig, raw_data.get("dataset")),
        training=_build_dataclass(TrainingConfig, raw_data.get("training")),
    )


__all__ = [
    "Config",
    "DatasetConfig",
    "GeneralConfig",
    "ModelConfig",
    "TrainingConfig",
    "load_config",
]
