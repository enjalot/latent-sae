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

    # -- K-annealing --
    # From chanind's autonomous SAE research: start training with a higher k and
    # linearly decrease to the target k. Gives the model an easier learning signal
    # early on — more features active means richer gradients for the decoder.
    # Similar in spirit to JumpReLU's learned sparsity (start loose, tighten).
    # See: lesswrong.com/posts/rbqJoxFZtae9x93mx
    k_anneal: bool = False
    """Enable k-annealing: linearly decrease k from k_anneal_start to k during training."""

    k_anneal_start: int = 0
    """Starting k for annealing. If 0, defaults to 4 * k."""

    k_anneal_pct: float = 0.3
    """Fraction of training to anneal k over (0.3 = anneal for first 30% of steps)."""


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

    # -- Tilted ERM --
    # Tilted Empirical Risk Minimization upweights high-loss samples by replacing
    # the mean loss with: (1/t) * log(mean(exp(t * loss_i))). With small t (~2e-3),
    # this gently focuses training on harder-to-reconstruct inputs, preventing the
    # SAE from only learning features for the most common/easy patterns.
    # Found effective by chanind's autonomous SAE research.
    # See: Li et al. "Tilted ERM" (NeurIPS 2021), lesswrong.com/posts/rbqJoxFZtae9x93mx
    tilted_erm_tilt: float = 0.0
    """Tilted ERM tilt parameter. 0 = disabled, ~2e-3 = gentle upweighting of hard samples."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    checkpoints_directory: str = "/checkpoints"

    log_to_wandb: bool = True
    wandb_project: str = "latent-sae"
    run_name: Union[str, None] = None
    wandb_log_frequency: int = 1
