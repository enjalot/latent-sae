"""
Experiment configuration system for SAE training.

Supports:
- YAML config files for reproducible experiments
- CLI overrides for quick iteration (dot-notation: sae.k=128)
- Config hashing for deduplication of sweep runs
- Cartesian sweep generation from parameter lists
- Adapter to existing TrainConfig / SaeConfig used by SaeTrainer

Modeled on latent-basemap/experiments/experiment_config.py.
"""

import copy
import hashlib
import itertools
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Where embeddings come from."""
    datasets: List[str] = field(default_factory=list)
    d_in: int = 768
    file_type: str = "npy"       # "npy" or "pt"
    shuffle: bool = True
    n_samples: Optional[int] = None  # None = use all, int = subsample (for benchmarks)
    random_seed: int = 42
    raw_dtype: str = "float32"   # element dtype for raw-memmap .npy (no header). Proper
                                 # .npy files auto-detect dtype and ignore this field.


@dataclass
class InfraConfig:
    """Infrastructure / hardware settings."""
    gpu_type: str = "a10g"       # "t4", "a10g", "a100_40gb"
    timeout_hours: int = 10


@dataclass
class LoggingConfig:
    """Where results go."""
    use_wandb: bool = True
    wandb_project: str = "latent-sae"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_group: Optional[str] = None
    wandb_log_frequency: int = 1
    results_dir: str = "experiments/results"
    save_model: bool = True


@dataclass
class ExperimentConfig:
    """
    Top-level experiment config.

    `sae` and `train` are plain dicts mapping to SaeConfig / TrainConfig fields.
    This avoids coupling to simple_parsing's Serializable while allowing full
    YAML round-tripping. Use `to_train_config()` to build what SaeTrainer expects.
    """
    name: str = "unnamed"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    sae: Dict[str, Any] = field(default_factory=lambda: {
        "sae_type": "topk",
        "expansion_factor": 32,
        "k": 64,
    })
    train: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 512,
        "grad_acc_steps": 4,
        "lr_warmup_steps": 1000,
        "use_amp": True,
        "save_every": 1000,
    })
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "data": asdict(self.data),
            "sae": dict(self.sae),
            "train": dict(self.train),
            "logging": asdict(self.logging),
            "infra": asdict(self.infra),
        }

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            data=DataConfig(**d.get("data", {})),
            sae=d.get("sae", {}),
            train=d.get("train", {}),
            logging=LoggingConfig(**d.get("logging", {})),
            infra=InfraConfig(**d.get("infra", {})),
        )

    def config_hash(self) -> str:
        """Short hash of config for deduplication."""
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:8]

    def run_dir(self) -> str:
        """Directory for this run's results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            self.logging.results_dir,
            f"{self.name}_{timestamp}_{self.config_hash()}",
        )

    def apply_overrides(self, overrides: dict):
        """Apply dot-notation overrides like {'sae.k': 128, 'train.batch_size': 1024}."""
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) != 2:
                raise ValueError(f"Override key must be section.param, got: {key}")

            section, param = parts

            # For dataclass sections, set attribute directly
            if section in ("data", "logging", "infra"):
                sub = getattr(self, section, None)
                if sub is not None and hasattr(sub, param):
                    current = getattr(sub, param)
                    if current is not None:
                        value = type(current)(value)
                    setattr(sub, param, value)
                else:
                    raise ValueError(f"Unknown config key: {key}")
            # For dict sections (sae, train), update the dict
            elif section in ("sae", "train"):
                d = getattr(self, section)
                if param in d:
                    current = d[param]
                    if current is not None:
                        value = type(current)(value)
                else:
                    # New key not in dict — try to infer type
                    try:
                        value = float(value)
                        if value == int(value) and "." not in str(value):
                            value = int(value)
                    except (ValueError, TypeError):
                        if value in ("true", "True"):
                            value = True
                        elif value in ("false", "False"):
                            value = False
                d[param] = value
            else:
                raise ValueError(f"Unknown config section: {section}")

    def to_train_config(self):
        """Build a TrainConfig that the existing SaeTrainer expects."""
        from latentsae.utils.config import SaeConfig, SaeType, TrainConfig, LRSchedule

        # Build SaeConfig from dict, handling enum conversion
        sae_dict = dict(self.sae)
        if "sae_type" in sae_dict:
            sae_dict["sae_type"] = SaeType(sae_dict["sae_type"])
        sae_cfg = SaeConfig(**sae_dict)

        # Build TrainConfig from dict, handling enum conversion
        train_dict = dict(self.train)
        if "lr_schedule" in train_dict:
            train_dict["lr_schedule"] = LRSchedule(train_dict["lr_schedule"])

        return TrainConfig(
            sae=sae_cfg,
            d_in=self.data.d_in,
            log_to_wandb=self.logging.use_wandb,
            wandb_project=self.logging.wandb_project,
            run_name=self.logging.wandb_run_name or self.name,
            wandb_log_frequency=self.logging.wandb_log_frequency,
            gpu_type=self.infra.gpu_type,
            **train_dict,
        )


def load_config(path: str, overrides: Optional[dict] = None) -> ExperimentConfig:
    """Load a YAML config file with optional CLI overrides."""
    config = ExperimentConfig.from_yaml(path)
    if overrides:
        config.apply_overrides(overrides)
    return config


def generate_sweep_configs(
    base_config: ExperimentConfig, sweep: Dict[str, list]
) -> List[ExperimentConfig]:
    """
    Generate configs for a parameter sweep (cartesian product).

    Example:
        sweep = {
            'sae.sae_type': ['topk', 'gated', 'jumprelu'],
            'sae.k': [32, 64],
        }
    Produces 6 configs (3 types x 2 k values).
    """
    keys = list(sweep.keys())
    values = list(sweep.values())
    configs = []

    for combo in itertools.product(*values):
        cfg = ExperimentConfig.from_dict(base_config.to_dict())
        overrides = dict(zip(keys, combo))
        cfg.apply_overrides(overrides)
        # Name the run after the swept parameters
        param_str = "_".join(f"{k.split('.')[-1]}{v}" for k, v in overrides.items())
        cfg.name = f"{base_config.name}__{param_str}"
        configs.append(cfg)

    return configs
