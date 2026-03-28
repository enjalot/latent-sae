"""
Sparse Autoencoder with support for multiple architecture variants:
- TopK: Standard top-k sparsity with ReLU pre-activations (Gao et al. 2024)
- Gated: Separate gating and magnitude paths (Rajamanoharan et al. 2024)
- JumpReLU: Learnable per-feature thresholds (Rajamanoharan et al. 2024)

Based on https://github.com/EleutherAI/sae
"""
import json
from typing import NamedTuple, Union, Optional
from pathlib import Path

import einops
import torch
from torch import Tensor, nn
from safetensors.torch import load_model, save_model
from huggingface_hub import snapshot_download

from .utils.config import SaeConfig, SaeType
from .utils.eleuther import decoder_impl


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""


class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        if cfg.sae_type == SaeType.GATED:
            # Gated SAE: separate gating and magnitude paths
            self.W_gate = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.W_mag = nn.Linear(d_in, self.num_latents, bias=False, device=device, dtype=dtype)
            self.r_mag = nn.Parameter(torch.ones(self.num_latents, device=device, dtype=dtype))
            self.W_gate.bias.data.zero_()
        else:
            # TopK and JumpReLU both use a single encoder
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.bias.data.zero_()

        if cfg.sae_type == SaeType.JUMPRELU:
            # Learnable per-feature thresholds
            self.log_threshold = nn.Parameter(
                torch.full((self.num_latents,), torch.tensor(cfg.jumprelu_init_threshold).log().item(),
                           device=device, dtype=dtype)
            )

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone() if hasattr(self, 'encoder') else self.W_gate.weight.data.clone()) if decoder else None
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @staticmethod
    def load_from_disk(
        path: Union[Path, str],
        device: Union[str, torch.device] = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            strict=decoder,
        )
        return sae

    @staticmethod
    def load_from_hub(
        name: str,
        k_expansion: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{k_expansion}/*" if k_expansion is not None else None,
            )
        )
        if k_expansion is not None:
            repo_path = repo_path / k_expansion
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    def save_to_disk(self, path: Union[Path, str]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        if hasattr(self, 'encoder'):
            return self.encoder.weight.device
        return self.W_gate.weight.device

    @property
    def dtype(self):
        if hasattr(self, 'encoder'):
            return self.encoder.weight.dtype
        return self.W_gate.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        """Compute pre-activations (before sparsity) by subtracting decoder bias and encoding."""
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)
        return nn.functional.relu(out)

    def encode_topk(self, x: Tensor) -> EncoderOutput:
        """TopK encoding: ReLU pre-activations then select top-k."""
        latents = self.pre_acts(x)
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def encode_gated(self, x: Tensor) -> EncoderOutput:
        """Gated SAE encoding: separate gate (which features) and magnitude (how much)."""
        sae_in = x.to(self.dtype) - self.b_dec
        # Gate determines which features fire
        gate_logits = self.W_gate(sae_in)
        # Magnitude determines activation strength, scaled by learnable r_mag
        mag = self.W_mag(sae_in) * self.r_mag.exp()

        # Select top-k by gate logits, use magnitude for activation values
        top_values, top_indices = gate_logits.topk(self.cfg.k, sorted=False)
        top_mag = mag.gather(-1, top_indices)
        # Gate with sigmoid, activation is gated magnitude (ReLU for non-negativity)
        top_acts = nn.functional.relu(top_mag * torch.sigmoid(top_values))
        return EncoderOutput(top_acts, top_indices)

    def encode_jumprelu(self, x: Tensor) -> EncoderOutput:
        """JumpReLU encoding: features fire only if pre-activation exceeds learnable threshold."""
        sae_in = x.to(self.dtype) - self.b_dec
        pre_acts = self.encoder(sae_in)

        threshold = self.log_threshold.exp()
        # Step function with straight-through estimator for gradients
        if self.training:
            # Smooth approximation of step function for gradient flow
            bandwidth = self.cfg.jumprelu_bandwidth
            mask = torch.sigmoid((pre_acts - threshold) / bandwidth)
        else:
            mask = (pre_acts > threshold).float()

        acts = pre_acts * mask

        # Return active features as sparse top-k-like output for compatibility
        # Select features with nonzero activations, capped at a reasonable k
        # For JumpReLU, the effective sparsity is learned, not fixed
        nonzero_count = (acts > 0).sum(dim=-1)
        max_k = min(self.cfg.k * 4, self.num_latents)
        top_acts, top_indices = acts.topk(min(max_k, acts.shape[-1]), sorted=False)

        # Trim to actual nonzero entries per row (keep at least 1)
        # For efficiency, we keep max_k but zeros won't affect the decode
        return EncoderOutput(top_acts, top_indices)

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode input using the configured architecture variant."""
        if self.cfg.sae_type == SaeType.GATED:
            return self.encode_gated(x)
        elif self.cfg.sae_type == SaeType.JUMPRELU:
            return self.encode_jumprelu(x)
        else:
            return self.encode_topk(x)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def forward(self, x: Tensor, dead_mask: Optional[Tensor] = None) -> ForwardOutput:
        # Encode
        if self.cfg.sae_type == SaeType.TOPK:
            pre_acts = self.pre_acts(x)
            top_acts, top_indices = pre_acts.topk(self.cfg.k, sorted=False)
        elif self.cfg.sae_type == SaeType.GATED:
            top_acts, top_indices = self.encode_gated(x)
            pre_acts = None  # Gated SAE doesn't use pre_acts for auxk
        elif self.cfg.sae_type == SaeType.JUMPRELU:
            top_acts, top_indices = self.encode_jumprelu(x)
            pre_acts = None
        else:
            raise ValueError(f"Unknown SAE type: {self.cfg.sae_type}")

        # Decode and compute residual
        sae_out = self.decode(top_acts, top_indices)
        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum()

        # AuxK loss for dead features (only supported for TopK)
        if dead_mask is not None and pre_acts is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = x.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        # JumpReLU sparsity penalty (L0 approximation)
        if self.cfg.sae_type == SaeType.JUMPRELU and self.training:
            l0_approx = (top_acts > 0).float().sum(dim=-1).mean()
            auxk_loss = self.cfg.sparsity_penalty * l0_approx

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        # Multi-TopK auxiliary loss (TopK only)
        if self.cfg.multi_topk and pre_acts is not None:
            mt_acts, mt_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            mt_out = self.decode(mt_acts, mt_indices)
            multi_topk_fvu = (mt_out - x).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
