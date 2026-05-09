"""Cross-SAE feature alignment.

For two SAEs trained on the same input space (ColBERT mxbai-edge-32m
token vectors), do they discover the same feature directions in
different parameterizations, or fundamentally different ones?

Method:
  1. Sample N corpus docs from TREC-COVID (cached)
  2. For each SAE, encode all doc tokens, pool to per-doc max
     activation per feature → matrix (n_docs, num_latents)
  3. Build a sparse activation matrix F per SAE
  4. For each feature in SAE A, compute cosine similarity to every
     feature in SAE B (over the same docs), find argmax → "best match"
  5. Report:
       - distribution of best-match correlations
       - how many features in A have correlation > 0.5 with some
         feature in B (we call those "aligned")
       - top examples of aligned feature pairs with their labels

Usage:
    python -m experiments.cross_sae_alignment \\
        --sae-a <17b dir> --sae-b <17r dir> \\
        --colbert-cache /data/embeddings/beir/trec-covid-mxbai-edge-32m \\
        --n-docs 2000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402
from experiments.maxsim_attribution import (  # noqa: E402
    find_sae_ckpt, load_packed,
)


@torch.no_grad()
def per_doc_feature_matrix(d_vecs: np.ndarray, d_offsets: np.ndarray,
                            doc_indices: list[int], sae: Sae,
                            device: str = "cuda",
                            agg: str = "max") -> np.ndarray:
    """Returns dense (n_docs, num_latents) per-doc max-activation matrix."""
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
        acts = sae_out.latent_acts.cpu().numpy()    # (n_tokens, k)
        idxs = sae_out.latent_indices.cpu().numpy() # (n_tokens, k)
        # Scatter into the row, keeping max per feature
        row = out[i]
        for t in range(idxs.shape[0]):
            for kk in range(idxs.shape[1]):
                f = int(idxs[t, kk])
                a = float(acts[t, kk])
                if agg == "max":
                    if a > row[f]:
                        row[f] = a
                elif agg == "sum":
                    row[f] += a
    return out


def best_match(F_a: np.ndarray, F_b: np.ndarray,
               batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each feature column in F_a, find the column in F_b with
    highest cosine similarity. Returns (matched_indices, best_corrs).

    F_a: (n_docs, n_a), F_b: (n_docs, n_b) — column-major: each column
    is a feature's activation vector across docs.
    """
    # l2-normalize columns
    norms_a = np.linalg.norm(F_a, axis=0, keepdims=True) + 1e-8  # (1, n_a)
    norms_b = np.linalg.norm(F_b, axis=0, keepdims=True) + 1e-8  # (1, n_b)
    Fa = F_a / norms_a
    Fb = F_b / norms_b

    n_a = Fa.shape[1]
    matched = np.zeros(n_a, dtype=np.int64)
    best_c = np.zeros(n_a, dtype=np.float32)
    # Stream to avoid building full n_a × n_b matrix
    for start in range(0, n_a, batch_size):
        end = min(start + batch_size, n_a)
        # (batch, n_docs) @ (n_docs, n_b) → (batch, n_b)
        sims = Fa[:, start:end].T @ Fb
        idx = sims.argmax(axis=1)
        bc = sims[np.arange(end - start), idx]
        matched[start:end] = idx
        best_c[start:end] = bc
    return matched, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-a", required=True)
    ap.add_argument("--sae-b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--colbert-cache", required=True)
    ap.add_argument("--n-docs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd_a = Path(args.sae_a); rd_b = Path(args.sae_b)
    sae_a = Sae.load_from_disk(find_sae_ckpt(rd_a), device=args.device)
    sae_b = Sae.load_from_disk(find_sae_ckpt(rd_b), device=args.device)
    print(f"SAE A ({args.name_a}): num_latents={sae_a.num_latents}")
    print(f"SAE B ({args.name_b}): num_latents={sae_b.num_latents}")

    cache = Path(args.colbert_cache)
    dv, do, d_ids = load_packed(cache, "corpus")
    rng = np.random.default_rng(args.seed)
    n_total = len(d_ids)
    if args.n_docs >= n_total:
        sampled_idx = list(range(n_total))
    else:
        sampled_idx = rng.choice(n_total, size=args.n_docs, replace=False).tolist()
    print(f"sampled {len(sampled_idx)} docs from {n_total}")

    print(f"\nencoding through {args.name_a}...")
    F_a = per_doc_feature_matrix(dv, do, sampled_idx, sae_a, device=args.device)
    print(f"  F_{args.name_a}: shape={F_a.shape}, nonzero ratio={(F_a > 0).mean():.4f}")

    print(f"\nencoding through {args.name_b}...")
    F_b = per_doc_feature_matrix(dv, do, sampled_idx, sae_b, device=args.device)
    print(f"  F_{args.name_b}: shape={F_b.shape}, nonzero ratio={(F_b > 0).mean():.4f}")

    # Filter to only "live" features (any activation in this sample) on each side
    live_a = (F_a > 0).any(axis=0)
    live_b = (F_b > 0).any(axis=0)
    print(f"\nlive in sample: {args.name_a}={live_a.sum()}, {args.name_b}={live_b.sum()}")
    F_a_live = F_a[:, live_a]
    F_b_live = F_b[:, live_b]
    a_live_ids = np.where(live_a)[0]
    b_live_ids = np.where(live_b)[0]

    # A → B match
    print(f"\nbest-match {args.name_a} → {args.name_b}...")
    matched_ab, bc_ab = best_match(F_a_live, F_b_live)
    # Translate back to global feature ids
    matched_ab_gids = b_live_ids[matched_ab]

    # B → A match
    print(f"best-match {args.name_b} → {args.name_a}...")
    matched_ba, bc_ba = best_match(F_b_live, F_a_live)
    matched_ba_gids = a_live_ids[matched_ba]

    # Distributions
    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    print(f"\nbest-match correlation distribution ({args.name_a} → {args.name_b}):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((bc_ab >= lo) & (bc_ab < hi)).sum()
        print(f"  [{lo:.2f}, {hi:.2f}): {n:>6} ({100*n/len(bc_ab):.1f}%)")

    print(f"\nbest-match correlation distribution ({args.name_b} → {args.name_a}):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((bc_ba >= lo) & (bc_ba < hi)).sum()
        print(f"  [{lo:.2f}, {hi:.2f}): {n:>6} ({100*n/len(bc_ba):.1f}%)")

    # Mutual: features where A→B match's B→A match returns to same A
    # (mutual nearest neighbors)
    a_to_b_local = matched_ab           # for each local A index, local B match
    b_to_a_local = matched_ba           # for each local B index, local A match
    # Mutual NN check
    mutual = 0
    mutual_corrs = []
    for la in range(len(a_live_ids)):
        lb = a_to_b_local[la]
        la_back = b_to_a_local[lb]
        if la_back == la:
            mutual += 1
            mutual_corrs.append(float(bc_ab[la]))
    print(f"\nmutual nearest neighbors (A↔B): {mutual} / {len(a_live_ids)} A features "
          f"({100*mutual/len(a_live_ids):.1f}%)")
    if mutual_corrs:
        print(f"  mean MNN correlation: {np.mean(mutual_corrs):.3f}")
        print(f"  >0.7 MNN pairs: {sum(1 for c in mutual_corrs if c > 0.7)}")
        print(f"  >0.5 MNN pairs: {sum(1 for c in mutual_corrs if c > 0.5)}")

    # Save top-K MNN pairs with descriptions
    labels_a = {}
    labels_b = {}
    if (rd_a / "feature_labels.json").exists():
        labels_a = json.loads((rd_a / "feature_labels.json").read_text())["labels"]
    if (rd_b / "feature_labels.json").exists():
        labels_b = json.loads((rd_b / "feature_labels.json").read_text())["labels"]

    mnn_pairs = []
    for la in range(len(a_live_ids)):
        lb = a_to_b_local[la]
        if b_to_a_local[lb] != la:
            continue
        gid_a = int(a_live_ids[la])
        gid_b = int(b_live_ids[lb])
        mnn_pairs.append({
            "a_feature": gid_a, "b_feature": gid_b,
            "correlation": float(bc_ab[la]),
            "a_judgment": labels_a.get(str(gid_a), {}).get("judgment", ""),
            "b_judgment": labels_b.get(str(gid_b), {}).get("judgment", ""),
            "a_description": labels_a.get(str(gid_a), {}).get("description", "")[:160],
            "b_description": labels_b.get(str(gid_b), {}).get("description", "")[:160],
        })
    mnn_pairs.sort(key=lambda x: -x["correlation"])

    # Print top 15 MNN pairs
    print(f"\nTop 15 mutual-NN feature pairs (correlation):")
    print(f"  {'corr':>5}  {args.name_a + '_fid':>8}  {args.name_b + '_fid':>8}  desc")
    for p in mnn_pairs[:15]:
        a_short = (p["a_description"] or "")[:55]
        b_short = (p["b_description"] or "")[:55]
        print(f"  {p['correlation']:>5.3f}  f{p['a_feature']:<7}  f{p['b_feature']:<7}  "
              f"[{p['a_judgment'][:4]}] {a_short}")
        print(f"        ↔ [{p['b_judgment'][:4]}] {b_short}")

    out_path = (Path(args.out) if args.out
                else Path(f"/data/embeddings/beir/cross-sae-{args.name_a}-vs-{args.name_b}.json"))
    out_path.write_text(json.dumps({
        "sae_a": str(rd_a), "sae_b": str(rd_b),
        "name_a": args.name_a, "name_b": args.name_b,
        "n_docs": len(sampled_idx),
        "live_a": int(live_a.sum()), "live_b": int(live_b.sum()),
        "mutual_nn_count": mutual,
        "mutual_nn_pct_a": 100 * mutual / len(a_live_ids),
        "mean_mnn_correlation": float(np.mean(mutual_corrs)) if mutual_corrs else 0.0,
        "mnn_pairs": mnn_pairs[:500],  # top 500 only to keep file size sane
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
