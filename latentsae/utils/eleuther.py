"""
Utilities from https://github.com/EleutherAI/sae - geometric median and decoder implementations.
"""
import os

import torch
from torch import Tensor


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median of `points`. Used for initializing decoder bias."""
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    for _ in range(max_iter):
        prev = guess
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        if torch.norm(guess - prev) < tol:
            break

    return guess


def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    """Fallback (non-Triton) implementation of sparse SAE decoder."""
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    """Triton-accelerated sparse SAE decoder."""
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode
