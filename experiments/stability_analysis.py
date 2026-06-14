"""Cross-seed feature stability for SAEs trained identically except for seed
(plan-sae-validation.md V3; protocol after Paulo & Belrose, arXiv 2501.16615).

For each pair of SAEs:
  - decoder rows unit-normalized; for each live feature in A, its max cosine to
    any feature in B (nearest-neighbour match). Report the fraction of A's live
    features with a match >= 0.7 and >= 0.5, and the mean NN cosine. This is the
    feature-overlap statistic; ~30% at 0.7 is typical for TopK on LLMs.
  - linear CKA between the two decoder matrices (permutation-invariant
    subspace-similarity; complements the 1-to-1 matching).

Usage:
  python -m experiments.stability_analysis \\
    --ckpts seed42:RUNDIR seed43:RUNDIR seed44:RUNDIR --device cuda
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


def _final_ckpt(run_dir: Path) -> Path:
    cands = [p for p in (run_dir / "checkpoints").glob("*")
             if p.is_dir() and (p / "cfg.json").exists()
             and not p.name.startswith("sae_step_")]
    if not cands:
        raise FileNotFoundError(f"no final checkpoint in {run_dir}")
    return cands[0]


def load_decoder(run_dir: Path, device: str):
    sae = Sae.load_from_disk(_final_ckpt(run_dir), device=device)
    W = sae.W_dec.detach().float()              # (num_latents, d)
    W = torch.nn.functional.normalize(W, dim=-1)
    return W


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    # X,Y: (n, d) feature-direction matrices, treated as n points in R^d.
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum()
    xx = (X.T @ X).pow(2).sum()
    yy = (Y.T @ Y).pow(2).sum()
    return float(xy / (xx.sqrt() * yy.sqrt() + 1e-12))


@torch.no_grad()
def nn_overlap(A: torch.Tensor, B: torch.Tensor, device: str,
               thresholds=(0.5, 0.7), block: int = 4096):
    """For each row of A, max cosine to any row of B (A,B unit-normalized)."""
    A = A.to(device); B = B.to(device)
    best = torch.empty(A.shape[0], device=device)
    for s in range(0, A.shape[0], block):
        sim = A[s:s + block] @ B.T          # (blk, |B|)
        best[s:s + block] = sim.max(dim=1).values
    out = {"mean_nn_cos": float(best.mean())}
    for t in thresholds:
        out[f"frac_ge_{t}"] = float((best >= t).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="tag:run_dir pairs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    decs = {}
    for spec in args.ckpts:
        tag, rd = spec.split(":", 1)
        decs[tag] = load_decoder(Path(rd), args.device)
        print(f"{tag}: W_dec {tuple(decs[tag].shape)}")

    results = {}
    for a, b in itertools.combinations(decs, 2):
        ov_ab = nn_overlap(decs[a], decs[b], args.device)
        ov_ba = nn_overlap(decs[b], decs[a], args.device)
        cka = linear_cka(decs[a].cpu(), decs[b].cpu())
        key = f"{a}__{b}"
        results[key] = {"a_to_b": ov_ab, "b_to_a": ov_ba, "cka": cka}
        print(f"\n{key}:")
        print(f"  {a}->{b}: mean_nn={ov_ab['mean_nn_cos']:.3f} "
              f"frac>=0.7={ov_ab['frac_ge_0.7']:.3f} frac>=0.5={ov_ab['frac_ge_0.5']:.3f}")
        print(f"  {b}->{a}: mean_nn={ov_ba['mean_nn_cos']:.3f} "
              f"frac>=0.7={ov_ba['frac_ge_0.7']:.3f} frac>=0.5={ov_ba['frac_ge_0.5']:.3f}")
        print(f"  decoder linear CKA: {cka:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
