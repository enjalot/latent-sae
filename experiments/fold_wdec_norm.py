"""Post-hoc fold of ||W_dec[i]|| into encoder weights.

For each latent i, let s[i] = ||W_dec[i, :]||_2. Apply:
    W_dec[i, :]        ← W_dec[i, :] / s[i]        (unit-norm decoder rows)
    encoder.weight[i, :] ← encoder.weight[i, :] * s[i]
    encoder.bias[i]      ← encoder.bias[i] * s[i]

Notes on invariance:
  - For pointwise nonlinearities (ReLU / JumpReLU / Gated), folding is
    exactly reconstruction-invariant: pre_act' = s[i] * pre_act, top_acts'
    = s[i] * top_acts, W_dec' = W_dec / s[i] → recon' == recon.
  - For TopK SAEs, the SELECTION is not invariant when s's differ: latents
    with larger ||W_dec|| are promoted in the top-k ranking. This is the
    interesting case — folding re-ranks latents by decoder contribution
    rather than raw encoder magnitude.

Writes a folded copy to <sae_dir>_folded/ (sibling of original dir),
mirroring the `checkpoints/<step>/` layout so existing eval scripts Just
Work by pointing at the folded run dir.

Usage:
  python -m experiments.fold_wdec_norm \\
      --run-dir experiments/results/<run>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


def find_sae_ckpt(run_dir: Path) -> Path | None:
    for p in sorted((run_dir / "checkpoints").glob("*"), key=lambda x: x.name):
        if p.is_dir() and (p / "cfg.json").exists():
            return p
    return None


@torch.no_grad()
def fold_decoder_norm(sae: Sae) -> dict:
    """Mutates the SAE in place. Returns a summary dict."""
    assert hasattr(sae, "encoder"), "fold supports TopK/JumpReLU encoder only"
    assert sae.W_dec is not None, "decoder must be present"

    W_dec = sae.W_dec.data                      # [num_latents, d_in]
    W_enc = sae.encoder.weight.data             # [num_latents, d_in]
    b_enc = sae.encoder.bias.data               # [num_latents]

    eps = torch.finfo(W_dec.dtype).eps
    norms = W_dec.norm(dim=1)                   # [num_latents]
    norms_safe = norms.clamp_min(eps)

    # Summary stats (before mutating)
    summary = {
        "num_latents": int(W_dec.shape[0]),
        "d_in": int(W_dec.shape[1]),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std().item()),
        "norm_median": float(norms.median().item()),
        "norm_p05": float(torch.quantile(norms, 0.05).item()),
        "norm_p95": float(torch.quantile(norms, 0.95).item()),
        "n_zero": int((norms < eps).sum().item()),
    }

    W_dec /= norms_safe.unsqueeze(1)
    W_enc *= norms_safe.unsqueeze(1)
    b_enc *= norms_safe
    return summary


@torch.no_grad()
def verify_pointwise_invariance(sae_before: Sae, sae_after: Sae,
                                n_samples: int = 2048, device: str = "cuda") -> dict:
    """For a fresh random batch, compare reconstruction pre vs post fold.

    For TopK SAEs, this is NOT expected to be exact — it measures how much
    the re-ranked top-k selection changes reconstruction.
    """
    d_in = sae_before.d_in
    torch.manual_seed(1234)
    x = torch.randn(n_samples, d_in, device=device) * 3.0
    out_before = sae_before(x).sae_out
    out_after = sae_after(x).sae_out
    mse = (out_before - out_after).pow(2).mean().item()
    rel = (out_before - out_after).norm(dim=1).mean().item() / (out_before.norm(dim=1).mean().item() + 1e-8)
    # Also: what fraction of top-k indices match?
    b = sae_before(x)
    a = sae_after(x)
    # both are Tensors; latent_indices shape [batch, k]
    match = 0
    total = 0
    for i in range(n_samples):
        ib = set(b.latent_indices[i].tolist())
        ia = set(a.latent_indices[i].tolist())
        match += len(ib & ia)
        total += len(ib)
    return {
        "recon_mse_before_vs_after": float(mse),
        "recon_rel_l2": float(rel),
        "topk_index_overlap": match / max(total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-suffix", default="_folded",
                    help="Sibling directory name: <run-dir><suffix>/")
    ap.add_argument("--verify", action="store_true",
                    help="Also measure reconstruction drift vs original on random samples")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = find_sae_ckpt(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    out_root = run_dir.parent / (run_dir.name + args.out_suffix)

    # Copy the run dir shell (config + metadata), leave activations/labels alone
    # since those are specific to the original and we want fresh versions.
    out_root.mkdir(parents=True, exist_ok=True)
    for fname in ["config.yaml", "results.json"]:
        src = run_dir / fname
        if src.exists():
            shutil.copy(src, out_root / fname)

    # Load SAE
    print(f"loading {ckpt}")
    sae = Sae.load_from_disk(ckpt, device=args.device)

    # Optionally: keep the pre-fold SAE around for invariance check
    sae_before = None
    if args.verify:
        sae_before = Sae.load_from_disk(ckpt, device=args.device)

    summary = fold_decoder_norm(sae)
    print(f"fold summary: min={summary['norm_min']:.4f} "
          f"p05={summary['norm_p05']:.4f} median={summary['norm_median']:.4f} "
          f"p95={summary['norm_p95']:.4f} max={summary['norm_max']:.4f}  "
          f"zero={summary['n_zero']}/{summary['num_latents']}")

    if args.verify and sae_before is not None:
        verify = verify_pointwise_invariance(sae_before, sae, device=args.device)
        print(f"verify: topk_overlap={verify['topk_index_overlap']:.3f}  "
              f"recon_mse={verify['recon_mse_before_vs_after']:.2e}  "
              f"rel_l2={verify['recon_rel_l2']:.3f}")
        summary.update(verify)

    # Save folded
    out_ckpt = out_root / "checkpoints" / ckpt.name
    out_ckpt.mkdir(parents=True, exist_ok=True)
    sae.save_to_disk(out_ckpt)
    (out_root / "fold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved folded checkpoint to {out_ckpt}")
    print(f"saved fold summary to {out_root / 'fold_summary.json'}")


if __name__ == "__main__":
    main()
