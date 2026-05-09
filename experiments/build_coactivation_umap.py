"""Feature co-activation UMAP.

Encode a sample of tokens through the SAE, build a (n_tokens, num_latents)
activation matrix, then UMAP the COLUMNS (each column is one feature's
activation vector across the token sample). Features that fire on
similar token sets cluster together — this is the geometry-free,
purely activation-driven feature similarity.

Two granularities:
  - "doc": pool max activation per feature per document (coarser)
  - "token": one row per token, top-k activations dense (finer)

For 22K-37K features we use cosine UMAP. To avoid the
"feature-fires-on-very-few-tokens" sparsity problem, we drop features
that fire on fewer than --min-tokens of the sample.

Usage:
    python -m experiments.build_coactivation_umap \\
        --sae-dir <run dir> \\
        --colbert-cache /data/embeddings/beir/trec-covid-mxbai-edge-32m \\
        --n-docs 3000 --granularity doc
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402
from experiments.maxsim_attribution import find_sae_ckpt, load_packed  # noqa: E402


@torch.no_grad()
def per_doc_max_activation(d_vecs: np.ndarray, d_offsets: np.ndarray,
                            doc_indices: list[int], sae: Sae,
                            device: str = "cuda") -> np.ndarray:
    """Per-doc max-activation matrix (n_docs, num_latents) — same as
    cross_sae_alignment but kept here to avoid import cycles."""
    sae.eval()
    n_docs = len(doc_indices)
    out = np.zeros((n_docs, sae.num_latents), dtype=np.float32)
    for i, idx in enumerate(doc_indices):
        s, e = int(d_offsets[idx]), int(d_offsets[idx + 1])
        if e <= s:
            continue
        d_tok = torch.from_numpy(np.ascontiguousarray(
            d_vecs[s:e], dtype=np.float32)).to(device)
        sae_out = sae(d_tok)
        acts = sae_out.latent_acts.cpu().numpy()
        idxs = sae_out.latent_indices.cpu().numpy()
        row = out[i]
        for t in range(idxs.shape[0]):
            for k in range(idxs.shape[1]):
                f = int(idxs[t, k])
                a = float(acts[t, k])
                if a > row[f]:
                    row[f] = a
    return out


@torch.no_grad()
def per_token_activation(d_vecs: np.ndarray, d_offsets: np.ndarray,
                          doc_indices: list[int], sae: Sae,
                          max_tokens: int = 50000,
                          device: str = "cuda") -> np.ndarray:
    """Per-token sparse activation matrix (n_tokens, num_latents) — but
    we keep only top-k entries per row (already done by SAE)."""
    sae.eval()
    rows = []  # list of (token_global_idx, feature_idxs, feature_acts)
    n_total = 0
    for i, idx in enumerate(doc_indices):
        s, e = int(d_offsets[idx]), int(d_offsets[idx + 1])
        if e <= s:
            continue
        d_tok = torch.from_numpy(np.ascontiguousarray(
            d_vecs[s:e], dtype=np.float32)).to(device)
        sae_out = sae(d_tok)
        acts = sae_out.latent_acts.cpu().numpy()      # (n_tok_doc, k)
        idxs = sae_out.latent_indices.cpu().numpy()   # (n_tok_doc, k)
        for t in range(idxs.shape[0]):
            rows.append((idxs[t], acts[t]))
            n_total += 1
            if n_total >= max_tokens:
                break
        if n_total >= max_tokens:
            break

    # Build dense matrix
    print(f"sampled {n_total} tokens; assembling sparse matrix...")
    out = np.zeros((n_total, sae.num_latents), dtype=np.float32)
    for i, (idxs, acts) in enumerate(rows):
        out[i, idxs] = acts
    return out


def umap_columns(M: np.ndarray, min_nonzero: int = 5,
                  n_neighbors: int = 30, min_dist: float = 0.1,
                  random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UMAP the COLUMNS of M (each column = one feature's activation
    vector across rows). Returns (Y, kept_columns)."""
    import umap
    from scipy.sparse import csr_matrix
    nz_per_col = (M != 0).sum(axis=0)
    keep = nz_per_col >= min_nonzero
    print(f"keeping {int(keep.sum())} / {M.shape[1]} features (>= {min_nonzero} nonzero rows)")
    M_kept = M[:, keep]
    # UMAP wants (n_samples, n_features); samples = features for us
    X = M_kept.T   # (n_kept, n_rows)
    print(f"UMAP cols: input {X.shape}, cosine metric...")
    t0 = time.monotonic()
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric="cosine",
                        random_state=random_state, n_jobs=1)
    Y = reducer.fit_transform(X)
    print(f"  done in {time.monotonic() - t0:.1f}s")
    kept_cols = np.where(keep)[0]
    return Y, kept_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--colbert-cache", required=True,
                    help="e.g. /data/embeddings/beir/trec-covid-mxbai-edge-32m")
    ap.add_argument("--granularity", default="doc", choices=["doc", "token"])
    ap.add_argument("--n-docs", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=80000,
                    help="(token granularity) cap on sampled tokens")
    ap.add_argument("--min-nonzero", type=int, default=5,
                    help="drop features with < this many nonzero activations in sample")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.sae_dir)
    sae = Sae.load_from_disk(find_sae_ckpt(rd), device=args.device)
    print(f"SAE: {rd.name}  num_latents={sae.num_latents}  d_in={sae.d_in}")

    cache = Path(args.colbert_cache)
    dv, do, d_ids = load_packed(cache, "corpus")
    rng = np.random.default_rng(args.seed)
    n_total = len(d_ids)
    sampled_idx = rng.choice(n_total, size=min(args.n_docs, n_total),
                              replace=False).tolist()
    print(f"sampled {len(sampled_idx)} docs")

    if args.granularity == "doc":
        M = per_doc_max_activation(dv, do, sampled_idx, sae, device=args.device)
    else:
        M = per_token_activation(dv, do, sampled_idx, sae,
                                  max_tokens=args.max_tokens, device=args.device)
    print(f"matrix shape: {M.shape}, nonzero ratio={(M != 0).mean():.4f}")

    Y, kept_cols = umap_columns(M, min_nonzero=args.min_nonzero,
                                  n_neighbors=args.n_neighbors,
                                  min_dist=args.min_dist)

    # Pull labels + scorer outputs
    labels_path = rd / "feature_labels.json"
    if labels_path.is_symlink():
        labels_path = labels_path.resolve()
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())["labels"]

    rows = []
    for i, fid in enumerate(kept_cols.tolist()):
        # n_nonzero in the sample (proxy for "how often does this fire")
        rows.append({
            "feature": int(fid),
            "judgment": labels.get(str(fid), {}).get("judgment", ""),
            "description": labels.get(str(fid), {}).get("description", ""),
            "coact_x": float(Y[i, 0]), "coact_y": float(Y[i, 1]),
            "n_nonzero_in_sample": int((M[:, fid] != 0).sum()),
            "max_act_in_sample": float(M[:, fid].max()),
        })
    out = (Path(args.out) if args.out
           else rd / f"feature_coactivation_{args.granularity}.parquet")
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({df.shape}, {out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
