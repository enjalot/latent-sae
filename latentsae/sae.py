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

        if cfg.sae_type == SaeType.BATCHTOPK or getattr(cfg, 'matryoshka_use_batchtopk', False):
            # Running estimate of the smallest accepted pre-activation during
            # training (populated via EMA in forward). Used at inference when
            # we don't have a full batch available.
            self.register_buffer("batchtopk_threshold",
                                 torch.tensor(cfg.batchtopk_threshold or 0.0,
                                              device=device, dtype=dtype))

        if cfg.sae_type == SaeType.MATRYOSHKA:
            sizes = [int(s) for s in cfg.matryoshka_sizes.split(",") if s.strip()]
            ks = [int(s) for s in cfg.matryoshka_ks.split(",") if s.strip()]
            if not sizes or len(sizes) != len(ks):
                raise ValueError(
                    f"Matryoshka requires matryoshka_sizes and matryoshka_ks "
                    f"with equal length; got sizes={sizes}, ks={ks}")
            if sizes[-1] != self.num_latents:
                raise ValueError(
                    f"largest matryoshka_sizes entry ({sizes[-1]}) must equal "
                    f"num_latents ({self.num_latents})")
            self._matryoshka_sizes = sizes
            self._matryoshka_ks = ks

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone() if hasattr(self, 'encoder') else self.W_gate.weight.data.clone()) if decoder else None
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
            # SAELens Apr-2024 recipe: rescale decoder rows to a fixed small norm
            # (default 0.1 for L1 SAEs) AFTER unit-norming, so init still has
            # consistent direction but a smaller scale. Skipped when == 0.
            init_norm = getattr(self.cfg, 'decoder_init_norm', 0.0) or 0.0
            if init_norm > 0:
                with torch.no_grad():
                    self.W_dec.data.mul_(init_norm)

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
        return self._encode_gated_with_k(x, self.cfg.k)

    def _encode_gated_with_k(self, x: Tensor, k: int) -> EncoderOutput:
        """Gated encoding with explicit k (supports k-annealing)."""
        sae_in = x.to(self.dtype) - self.b_dec
        # Gate determines which features fire
        gate_logits = self.W_gate(sae_in)
        # Magnitude determines activation strength, scaled by learnable r_mag
        mag = self.W_mag(sae_in) * self.r_mag.exp()

        # Select top-k by gate logits, use magnitude for activation values
        top_values, top_indices = gate_logits.topk(k, sorted=False)
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

    def encode_lista(self, x: Tensor, k: Optional[int] = None) -> EncoderOutput:
        """LISTA encoding: lateral inhibition via decoder weight correlations, then top-k.

        Instead of learning a separate S matrix (which would be num_latents^2 params),
        we compute inhibition from the decoder weights on-the-fly. The idea: if two
        features have similar decoder vectors, they represent similar concepts and should
        compete. We compute this only among the top-4k candidates to keep it tractable.

        Based on Gregor & LeCun 2010 (Learned ISTA) adapted for embedding SAEs.
        """
        k = k or self.cfg.k
        eta = self.cfg.lista_eta
        candidate_k = min(4 * k, self.num_latents)

        sae_in = x.to(self.dtype) - self.b_dec
        pre = nn.functional.relu(self.encoder(sae_in))  # [batch, num_latents]

        for _ in range(self.cfg.lista_steps):
            # Select candidates (top-4k by activation)
            cand_acts, cand_idx = pre.topk(candidate_k, sorted=False)  # [batch, 4k]

            # Get decoder vectors for candidates
            cand_dec = self.W_dec[cand_idx]  # [batch, 4k, d_in]

            # Pairwise correlations among candidates (decoder dot products)
            # [batch, 4k, 4k]
            corr = torch.bmm(cand_dec, cand_dec.transpose(-1, -2))

            # Zero self-correlations (don't self-inhibit)
            corr.diagonal(dim1=-2, dim2=-1).zero_()

            # Apply inhibition: subtract correlated activations
            inhibition = torch.bmm(corr, cand_acts.unsqueeze(-1)).squeeze(-1)  # [batch, 4k]
            cand_acts = nn.functional.relu(cand_acts - eta * inhibition)

            # Write back into full pre-activation tensor for next iteration
            if self.cfg.lista_steps > 1:
                pre = pre.scatter(-1, cand_idx, cand_acts)

        # Final top-k selection from inhibited candidates
        final_acts, final_local_idx = cand_acts.topk(k, sorted=False)  # [batch, k]
        final_indices = cand_idx.gather(-1, final_local_idx)  # [batch, k]

        return EncoderOutput(final_acts, final_indices)

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode input using the configured architecture variant (inference)."""
        if self.cfg.sae_type == SaeType.GATED:
            return self.encode_gated(x)
        elif self.cfg.sae_type == SaeType.JUMPRELU:
            return self.encode_jumprelu(x)
        elif self.cfg.sae_type == SaeType.LISTA:
            return self.encode_lista(x)
        elif self.cfg.sae_type == SaeType.BATCHTOPK:
            top_acts, top_indices, _ = self.encode_batchtopk_inference(x, self.cfg.k)
            return EncoderOutput(top_acts, top_indices)
        elif self.cfg.sae_type == SaeType.MATRYOSHKA:
            # At inference, use the largest level's features (full SAE behavior)
            latents = self.pre_acts(x)
            return EncoderOutput(*latents.topk(self._matryoshka_ks[-1], sorted=False))
        else:
            return self.encode_topk(x)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def encode_batchtopk(self, x: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
        """BatchTopK: pick the top (k * batch_size) scores across the whole batch.

        Returns (top_acts, top_indices, pre_acts) with shapes (B, k) matching
        TopK's interface — samples with fewer-than-k selections are zero-padded.
        During training we also update an EMA of the threshold (smallest
        accepted pre-activation) for inference use.
        """
        pre_acts = self.pre_acts(x)
        B, N = pre_acts.shape
        total = B * k
        # Flat top-(B*k); pre_acts is relu'd in pre_acts()
        flat = pre_acts.reshape(-1)
        vals, flat_idx = flat.topk(total, sorted=False)
        # Scatter back per-sample: row = flat_idx // N, col = flat_idx % N
        rows = flat_idx // N
        cols = flat_idx % N
        # Build (B, k) by ranking flat values within each row
        # Strategy: create (B, N) zero mask then place; select per-row top-k.
        mask = torch.zeros_like(pre_acts)
        mask[rows, cols] = vals
        top_acts, top_indices = mask.topk(k, sorted=False)
        # Filter padding (rows with fewer accepted than k have zero top_acts for extras)
        if self.training:
            min_accepted = vals.min()
            ema = self.cfg.batchtopk_threshold_beta
            self.batchtopk_threshold.mul_(ema).add_(min_accepted.detach() * (1 - ema))
        return top_acts, top_indices, pre_acts

    def encode_batchtopk_inference(self, x: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
        """At inference we don't have a batch — apply learned threshold, then cap at k."""
        pre_acts = self.pre_acts(x)
        threshold = self.batchtopk_threshold.to(pre_acts.dtype)
        gated = torch.where(pre_acts >= threshold, pre_acts,
                            pre_acts.new_zeros(()))
        # Cap at k active per sample for safety
        top_acts, top_indices = gated.topk(k, sorted=False)
        return top_acts, top_indices, pre_acts

    def forward(self, x: Tensor, dead_mask: Optional[Tensor] = None, k_override: Optional[int] = None) -> ForwardOutput:
        # k_override allows the trainer to dynamically change k (used for k-annealing)
        k = k_override or self.cfg.k

        # Matryoshka has its own forward: multi-level reconstruction losses
        if self.cfg.sae_type == SaeType.MATRYOSHKA:
            return self._forward_matryoshka(x, dead_mask)

        # Encode
        if self.cfg.sae_type == SaeType.TOPK:
            pre_acts = self.pre_acts(x)
            top_acts, top_indices = pre_acts.topk(k, sorted=False)
        elif self.cfg.sae_type == SaeType.BATCHTOPK:
            if self.training:
                top_acts, top_indices, pre_acts = self.encode_batchtopk(x, k)
            else:
                top_acts, top_indices, pre_acts = self.encode_batchtopk_inference(x, k)
        elif self.cfg.sae_type == SaeType.LISTA:
            pre_acts = self.pre_acts(x)  # needed for auxk
            top_acts, top_indices = self.encode_lista(x, k=k)
        elif self.cfg.sae_type == SaeType.GATED:
            top_acts, top_indices = self._encode_gated_with_k(x, k)
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
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
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
            mt_acts, mt_indices = pre_acts.topk(4 * k, sorted=False)
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

    def _forward_matryoshka(self, x: Tensor,
                            dead_mask: Optional[Tensor] = None) -> ForwardOutput:
        """Matryoshka forward: compute reconstruction loss at each nesting level.

        At each level i with latent count n_i and sparsity k_i:
          - pick top-k_i activations among pre_acts[:, :n_i]
          - decode with W_dec[:n_i, :]
          - accumulate FVU
        Primary sae_out / top_acts / top_indices reported for the LARGEST level
        so downstream eval treats this like a standard SAE.
        """
        sizes = self._matryoshka_sizes
        ks = self._matryoshka_ks
        pre_acts = self.pre_acts(x)       # (B, N_total)
        total_variance = (x - x.mean(0)).pow(2).sum()
        use_btk = getattr(self.cfg, 'matryoshka_use_batchtopk', False)

        def _select(sub: Tensor, lvl_k: int):
            """Return (lvl_acts, lvl_idx) of shape (B, lvl_k).

            When use_btk: pick top (B * lvl_k) pre_acts across the batch,
            then project back to a per-sample (B, lvl_k) tensor with zero
            padding for samples that received fewer than lvl_k. Matches
            Sae.encode_batchtopk but operates on a matryoshka level's
            pre_acts slice.
            """
            if not use_btk:
                return sub.topk(lvl_k, sorted=False)
            B, N = sub.shape
            total = B * lvl_k
            flat = sub.reshape(-1)
            vals, flat_idx = flat.topk(total, sorted=False)
            rows = flat_idx // N
            cols = flat_idx % N
            mask = torch.zeros_like(sub)
            mask[rows, cols] = vals
            return mask.topk(lvl_k, sorted=False)

        per_level_fvu = []
        for lvl_n, lvl_k in zip(sizes, ks):
            sub = pre_acts[:, :lvl_n]
            lvl_acts, lvl_idx = _select(sub, lvl_k)
            # Decode using the first lvl_n decoder rows
            y = decoder_impl(lvl_idx, lvl_acts.to(self.dtype),
                             self.W_dec[:lvl_n].mT)
            recon = y + self.b_dec
            e = recon - x
            per_level_fvu.append(e.pow(2).sum() / total_variance)

        # Track BatchTopK threshold for inference (EMA of smallest accepted
        # pre-act at the largest level during training — matches standalone
        # BatchTopK path).
        if use_btk and self.training:
            largest_n, largest_k = sizes[-1], ks[-1]
            total = pre_acts.shape[0] * largest_k
            min_accepted = pre_acts[:, :largest_n].reshape(-1).topk(total).values.min()
            ema = self.cfg.batchtopk_threshold_beta
            if hasattr(self, 'batchtopk_threshold'):
                self.batchtopk_threshold.mul_(ema).add_(min_accepted.detach() * (1 - ema))

        # The primary output uses the LARGEST level
        largest_n, largest_k = sizes[-1], ks[-1]
        top_acts, top_indices = _select(pre_acts[:, :largest_n], largest_k) if use_btk else pre_acts.topk(largest_k, sorted=False)
        sae_out = self.decode(top_acts, top_indices)

        # Use mean FVU across levels as the training loss surrogate
        fvu = torch.stack(per_level_fvu).mean()
        multi_topk_fvu = sae_out.new_tensor(0.0)
        auxk_loss = sae_out.new_tensor(0.0)
        return ForwardOutput(sae_out, top_acts, top_indices, fvu, auxk_loss,
                             multi_topk_fvu)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if self.W_dec.grad is None:
            return

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
