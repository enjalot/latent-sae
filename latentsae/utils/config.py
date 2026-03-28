from dataclasses import dataclass
from enum import Enum
from typing import Union

from simple_parsing import Serializable


class SaeType(str, Enum):
    """SAE architecture type."""
    TOPK = "topk"
    GATED = "gated"
    JUMPRELU = "jumprelu"


class LRSchedule(str, Enum):
    """Learning rate schedule type."""
    LINEAR = "linear"
    COSINE = "cosine"


@dataclass
class SaeConfig(Serializable):
    """Configuration for the sparse autoencoder architecture."""

    sae_type: SaeType = SaeType.TOPK
    """SAE architecture variant: topk, gated, or jumprelu."""

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features (for topk and gated variants)."""

    multi_topk: bool = False
    """Use Multi-TopK auxiliary loss (topk variant only)."""

    jumprelu_init_threshold: float = 0.001
    """Initial threshold for JumpReLU activation."""

    jumprelu_bandwidth: float = 0.001
    """Bandwidth for the straight-through estimator in JumpReLU."""

    sparsity_penalty: float = 1e-3
    """L0 sparsity penalty coefficient for JumpReLU."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    d_in: int = 768
    """Dimensions of input embeddings."""

    batch_size: int = 8
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: Union[float, None] = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_schedule: LRSchedule = LRSchedule.COSINE
    """Learning rate schedule: linear warmup+decay or cosine annealing."""

    lr_warmup_steps: int = 1000
    """Number of warmup steps for learning rate schedule."""

    use_amp: bool = True
    """Use automatic mixed precision (float16) for training."""

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    checkpoints_directory: str = "/checkpoints"

    log_to_wandb: bool = True
    wandb_project: str = "latent-sae"
    run_name: Union[str, None] = None
    wandb_log_frequency: int = 1
