"""Investigate the k=16/exp=2 retrieval anomaly.

That config has FVU 0.025 (middle-of-pack reconstruction) but nDCG@10 = 0.001
(catastrophic retrieval). Hypothesis: the reconstruction preserves
squared-error magnitude but rotates vectors in a way that breaks the
query-doc cosine geometry ColBERT MaxSim depends on.

Diagnostics computed here:
  - per-token cosine sim between raw and reconstructed
  - avg query-token vs doc-token MaxSim raw vs reconstructed on SciFact
  - effective rank of W_dec (tells us if reconstruction lives in a
    very low-dim subspace, which could explain the collapse)

Runs on CPU so it doesn't contend with GPU training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


CONFIGS = [
    ("k=8/exp=2 (clean winner, nDCG 0.245)",
     "experiments/results/colbert_mxbai_phase4__k8_expansion_factor2_20260421_004122_9206a7db"),
    ("k=16/exp=2 (ANOMALY, nDCG 0.001)",
     "experiments/results/colbert_mxbai_phase4__k16_expansion_factor2_20260421_005309_85b0646c"),
    ("k=24/exp=2 (nDCG 0.685)",
     "experiments/results/colbert_mxbai_phase4__k24_expansion_factor2_20260421_010429_c15ed723"),
    ("k=32/exp=2 (nDCG 0.698)",
     "experiments/results/colbert_mxbai_phase5__k32_20260421_013157_4f8bb47f"),
    ("k=32/exp=16 (Phase 1, nDCG 0.709)",
     "experiments/results/colbert_mxbai_phase1__k32_expansion_factor32_20260420_224221_4a8e2b8e"),
]


def effective_rank(W: torch.Tensor) -> float:
    """Shannon entropy of normalized singular values (in nats -> exp)."""
    s = torch.linalg.svdvals(W.float().cpu())
    s = s / s.sum()
    s = s[s > 1e-12]
    return float(torch.exp(-(s * s.log()).sum()))


def find_checkpoint(run_dir: Path) -> Path | None:
    return next((p for p in (run_dir / "checkpoints").glob("*")
                 if p.is_dir() and (p / "cfg.json").exists()), None)


@torch.no_grad()
def cpu_forward(sae: Sae, x: torch.Tensor) -> torch.Tensor:
    """Recon forward that avoids the triton-GPU-only decoder kernel."""
    pre = torch.nn.functional.relu(sae.encoder(x - sae.b_dec))
    top_vals, top_idx = pre.topk(sae.cfg.k, sorted=False)
    # Gather decoder rows for selected indices: (B, k, d_in)
    dec = sae.W_dec[top_idx]
    recon = (top_vals.unsqueeze(-1) * dec).sum(dim=1) + sae.b_dec
    return recon


@torch.no_grad()
def analyze(name: str, run_dir: str, query_vecs: np.ndarray, doc_vecs: np.ndarray):
    rd = Path(run_dir)
    ckpt = find_checkpoint(rd)
    if ckpt is None:
        print(f"SKIP {name}: no checkpoint")
        return {}
    sae = Sae.load_from_disk(ckpt, device="cpu")
    sae.eval()

    # Reconstruct a sample on CPU via manual matmul
    q_t = torch.from_numpy(query_vecs.astype(np.float32))
    d_t = torch.from_numpy(doc_vecs.astype(np.float32))
    q_r = cpu_forward(sae, q_t)
    d_r = cpu_forward(sae, d_t)

    # Cosine sim per token between raw and recon
    def per_row_cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    q_cos = per_row_cos(q_t, q_r).numpy()
    d_cos = per_row_cos(d_t, d_r).numpy()

    # Raw vs reconstructed max-pair similarity across tokens (mini maxsim on a sample)
    # Compute an all-pairs sim: (nq, nd)
    def maxsim(Qs, Ds, nq=30, nd=200):
        Qs = Qs[:nq]; Ds = Ds[:nd]
        sims = Qs @ Ds.T
        return sims.max(dim=-1).values.mean().item()

    raw_maxsim = maxsim(q_t, d_t)
    recon_maxsim = maxsim(q_r, d_r)

    # Decoder effective rank
    eff_rank = effective_rank(sae.W_dec)

    # Check if reconstruction collapses to small subspace
    # Sample 2000 query + doc tokens' reconstructions and compute rank
    sample = torch.cat([q_r[:1000], d_r[:1000]], dim=0)
    eff_rank_recon = effective_rank(sample)

    return {
        "name": name,
        "num_latents": sae.num_latents,
        "sae_type": sae.cfg.sae_type.value,
        "query_cos_raw_vs_recon_mean": float(q_cos.mean()),
        "query_cos_raw_vs_recon_min":  float(q_cos.min()),
        "doc_cos_raw_vs_recon_mean":   float(d_cos.mean()),
        "doc_cos_raw_vs_recon_min":    float(d_cos.min()),
        "raw_sample_maxsim":      float(raw_maxsim),
        "recon_sample_maxsim":    float(recon_maxsim),
        "W_dec_effective_rank":   float(eff_rank),
        "recon_effective_rank":   float(eff_rank_recon),
    }


def main():
    # Load cached SciFact ColBERT embeddings
    cache = Path("/data/embeddings/beir/scifact-mxbai-edge-32m")
    q_vecs = np.load(cache / "queries_vectors.npy", mmap_mode="r")
    d_vecs = np.load(cache / "corpus_vectors.npy", mmap_mode="r")
    print(f"loaded {q_vecs.shape[0]:,} query tokens + {d_vecs.shape[0]:,} doc tokens")

    # Sample a bounded set for analysis
    q_sample = np.asarray(q_vecs[:2000])
    d_sample = np.asarray(d_vecs[:5000])

    rows = []
    for name, rd in CONFIGS:
        print(f"\n== {name} ==")
        r = analyze(name, rd, q_sample, d_sample)
        rows.append(r)
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k:<32} {v:.4f}")
            else:
                print(f"  {k:<32} {v}")

    out = Path("/data/anomaly_report.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")

    # Pretty table
    print(f"\n{'config':50}  {'lat':>5}  {'q_cos_μ':>8} {'d_cos_μ':>8}  "
          f"{'raw_maxsim':>11} {'recon_maxsim':>13}  {'rank_Wdec':>10} {'rank_recon':>11}")
    for r in rows:
        if not r:
            continue
        print(f"{r['name']:50}  {r['num_latents']:>5}  "
              f"{r['query_cos_raw_vs_recon_mean']:>8.4f} {r['doc_cos_raw_vs_recon_mean']:>8.4f}  "
              f"{r['raw_sample_maxsim']:>11.4f} {r['recon_sample_maxsim']:>13.4f}  "
              f"{r['W_dec_effective_rank']:>10.2f} {r['recon_effective_rank']:>11.2f}")


if __name__ == "__main__":
    main()
