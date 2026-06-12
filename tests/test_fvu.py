"""FVU consistency: training-time FVU (sae.py) must equal post-hoc eval FVU
(eval_core_metrics.py) on the same data.

Training computes  fvu = sum(err^2) / sum((x - x.mean(0))^2)  per batch.
The eval script streams per-dimension sums over sub-batches; on a single
eval set the two must agree to float tolerance.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402
from latentsae.utils.config import SaeConfig, SaeType  # noqa: E402
from experiments.eval_core_metrics import eval_checkpoint  # noqa: E402

D_IN = 32
N = 512


def _make_sae(sae_type: SaeType) -> Sae:
    torch.manual_seed(0)
    cfg = SaeConfig(sae_type=sae_type, expansion_factor=4, k=8)
    if sae_type == SaeType.MATRYOSHKA:
        cfg.matryoshka_sizes = "32,128"
        cfg.matryoshka_ks = "4,8"
    return Sae(D_IN, cfg, device="cpu", dtype=torch.float32)


def _data() -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    # Non-zero per-dimension means so global-mean and per-dim-mean variance
    # actually differ — this is what the old eval formula got wrong.
    x = torch.randn(N, D_IN, generator=g) + torch.linspace(-2, 2, D_IN)
    return x


@pytest.mark.parametrize("sae_type", [SaeType.TOPK, SaeType.BATCHTOPK])
def test_eval_fvu_matches_training_fvu(tmp_path, sae_type):
    sae = _make_sae(sae_type)
    sae.eval()
    x = _data()

    with torch.no_grad():
        train_fvu = sae(x).fvu.item()

    ckpt = tmp_path / "ckpt"
    sae.save_to_disk(ckpt)
    m = eval_checkpoint(ckpt, x, batch_size=100, device="cpu")

    assert m["fvu"] == pytest.approx(train_fvu, rel=1e-4), (
        f"eval FVU {m['fvu']:.6f} != training FVU {train_fvu:.6f}"
    )


def test_per_dim_centering_is_the_training_denominator():
    x = _data()
    per_dim = (x - x.mean(0)).pow(2).sum().item()
    n = x.shape[0]
    x_sum = x.double().sum(dim=0)
    x_sq_sum = (x.double() ** 2).sum(dim=0)
    streamed = (x_sq_sum - x_sum.pow(2) / n).sum().item()
    assert streamed == pytest.approx(per_dim, rel=1e-6)
    # And the old (wrong) global-scalar formula really is different on this data
    n_elem = x.numel()
    global_var = (x.pow(2).sum() / n_elem - (x.sum() / n_elem) ** 2).item() * n_elem
    assert abs(global_var - per_dim) / per_dim > 0.1


@pytest.mark.parametrize(
    "sae_type",
    [SaeType.TOPK, SaeType.BATCHTOPK, SaeType.GATED, SaeType.JUMPRELU,
     SaeType.MATRYOSHKA],
)
def test_forward_smoke(sae_type):
    sae = _make_sae(sae_type)
    sae.eval()
    with torch.no_grad():
        out = sae(_data())
    assert out.sae_out.shape == (N, D_IN)
    assert torch.isfinite(out.fvu)
