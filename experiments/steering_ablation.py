"""Causal steering test: ablate or amplify a single SAE feature on
candidate documents and measure how it shifts the ColBERT MaxSim rank
of the gold-relevant document.

Setup per (query, feature, mode):
  1. Take top-K candidate documents by raw ColBERT MaxSim (precomputed)
  2. For each candidate: SAE-encode its tokens, modify the target
     feature's activation (zero or 2× depending on mode), SAE-decode
  3. Recompute MaxSim(query, reconstructed_doc) with raw-query tokens
  4. Re-sort the top-K, find the gold doc's new rank
  5. Report: baseline rank, new rank, score delta, MaxSim distribution

Backbone features (f175, f145, etc) should produce small rank shifts
because they fire on every candidate doc roughly equally — ablation
affects all uniformly. Query-distinctive features should produce
visible rank shifts because they're concentrated on specific docs.

Usage:
    python -m experiments.steering_ablation \\
        --sae-dir <17r dir> \\
        --colbert-cache /data/embeddings/beir/trec-covid-mxbai-edge-32m \\
        --qid 8 --did 8cw3bjxh --feature 4184 --mode ablate \\
        --top-k 100
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
    get_query_tokens, get_doc_tokens, maxsim_with_indices,
)


@torch.no_grad()
def sae_encode_decode_modified(d_tokens: np.ndarray, sae: Sae,
                                feature_id: int, multiplier: float,
                                device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """SAE-encode doc tokens, modify a single feature's activation by the
    given multiplier (at the LARGEST matryoshka level), and return both the
    UNMODIFIED reconstruction (full matryoshka) and the modified one.

    The modification subtracts (1 - multiplier) × (feature f's contribution
    at the largest level) from the full reconstruction. So:
      multiplier=0.0 → ablate: subtract f's contribution
      multiplier=2.0 → amplify: add 1× more of f's contribution
    Note: this only edits the largest-level slice of the matryoshka. For
    features that also fire at smaller levels, the modification is partial.
    """
    sae.eval()
    batch = torch.from_numpy(np.ascontiguousarray(d_tokens, dtype=np.float32)).to(device)
    out = sae(batch)
    # full reconstruction including matryoshka aggregation
    recon_full = out.sae_out                          # (n, d_in)
    acts = out.latent_acts                            # (n, k_largest)
    idxs = out.latent_indices                         # (n, k_largest)

    # For each token, get the activation of feature_id (0 if not in top-k)
    mask = (idxs == feature_id).float()               # (n, k)
    f_acts = (acts * mask).sum(dim=1)                 # (n,)

    # Contribution of feature_id at largest level: f_acts × W_dec[feature_id]
    w_row = sae.W_dec[feature_id]                     # (d_in,)
    contribution = f_acts.unsqueeze(-1) * w_row        # (n, d_in)

    # Modified = full - (1 - multiplier) × contribution
    recon_modified = recon_full - (1.0 - multiplier) * contribution
    return recon_full.cpu().numpy(), recon_modified.cpu().numpy()


@torch.no_grad()
def maxsim_score(q_tokens: np.ndarray, d_tokens: np.ndarray,
                 device: str = "cuda") -> float:
    Q = torch.from_numpy(np.ascontiguousarray(q_tokens, dtype=np.float32)).to(device)
    D = torch.from_numpy(np.ascontiguousarray(d_tokens, dtype=np.float32)).to(device)
    return float((Q @ D.T).max(dim=1).values.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--colbert-cache", required=True)
    ap.add_argument("--colbert-topk-cache",
                    default="/data/embeddings/beir/trec-covid-mxbai-edge-32m/colbert_topk.npz")
    ap.add_argument("--qid", required=True)
    ap.add_argument("--did", required=True,
                    help="gold doc id whose rank we track")
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--mode", default="ablate", choices=["ablate", "amplify", "double", "halve"])
    ap.add_argument("--top-k", type=int, default=100,
                    help="Number of top ColBERT candidate docs to re-rank under ablation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    multiplier = {"ablate": 0.0, "amplify": 2.0, "double": 2.0, "halve": 0.5}[args.mode]

    rd = Path(args.sae_dir)
    ckpt = find_sae_ckpt(rd)
    sae = Sae.load_from_disk(ckpt, device=args.device)

    cache = Path(args.colbert_cache)
    qv, qo, q_ids = load_packed(cache, "queries")
    dv, do, d_ids = load_packed(cache, "corpus")

    # Top-k candidate doc indices for this query (cached as top-k matrix)
    npz = np.load(args.colbert_topk_cache, allow_pickle=True)
    topk_mat = npz["top_k"]; cb_q_ids = list(npz["q_ids"]); cb_d_ids = list(npz["d_ids"])
    # We need to align our q_ids/d_ids with the topk cache's
    qi = cb_q_ids.index(args.qid)
    candidate_dids = [cb_d_ids[j] for j in topk_mat[qi, :args.top_k]]
    if args.did not in candidate_dids:
        print(f"WARN: gold doc {args.did} not in top-{args.top_k} candidates — "
              f"will append it to the candidate set so we can track its rank")
        candidate_dids = [args.did] + candidate_dids

    # Load query tokens once
    q_tokens = get_query_tokens(qv, qo, q_ids, args.qid)

    # Compute baseline MaxSim using RAW doc vectors for each candidate
    baseline = []
    for did in candidate_dids:
        d_tokens = get_doc_tokens(dv, do, d_ids, did)
        baseline.append((did, maxsim_score(q_tokens, d_tokens, device=args.device)))
    baseline.sort(key=lambda x: -x[1])
    baseline_rank = next((i for i, (d, _) in enumerate(baseline) if d == args.did), -1)

    # Now compute SAE-reconstructed scores for each candidate, with the
    # specified feature modified
    modified = []
    sae_baseline = []  # SAE-reconstructed without modification
    for did in candidate_dids:
        d_tokens = get_doc_tokens(dv, do, d_ids, did)
        recon_base, recon_mod = sae_encode_decode_modified(
            d_tokens, sae, args.feature, multiplier, device=args.device)
        sae_baseline.append((did, maxsim_score(q_tokens, recon_base,
                                                device=args.device)))
        modified.append((did, maxsim_score(q_tokens, recon_mod, device=args.device)))

    sae_baseline.sort(key=lambda x: -x[1])
    modified.sort(key=lambda x: -x[1])

    sae_baseline_rank = next((i for i, (d, _) in enumerate(sae_baseline) if d == args.did), -1)
    modified_rank = next((i for i, (d, _) in enumerate(modified) if d == args.did), -1)

    # Score deltas
    base_score = next(s for d, s in sae_baseline if d == args.did)
    mod_score = next(s for d, s in modified if d == args.did)

    # Print
    print(f"\n=== Steering test ===")
    print(f"qid={args.qid}, gold did={args.did}, feature={args.feature}, mode={args.mode}")
    print(f"top-{args.top_k} candidates, multiplier={multiplier}")
    print(f"\nGold doc rank:")
    print(f"  raw ColBERT MaxSim:        {baseline_rank}")
    print(f"  SAE recon (no mod):        {sae_baseline_rank}")
    print(f"  SAE recon ({args.mode} f{args.feature}): {modified_rank}")
    print(f"\nGold doc score:")
    print(f"  SAE recon (no mod):        {base_score:.3f}")
    print(f"  SAE recon ({args.mode} f{args.feature}): {mod_score:.3f}  "
          f"({mod_score - base_score:+.3f})")

    # How many candidates moved rank?
    base_order = [d for d, _ in sae_baseline]
    mod_order = [d for d, _ in modified]
    movers = sum(1 for i, d in enumerate(mod_order) if base_order[i] != d)
    print(f"\nrank shifts in top-{args.top_k}: {movers} positions changed")

    # Print top-5 by new ranking
    print(f"\nNew top-5 under {args.mode}:")
    for i in range(min(5, len(modified))):
        d, s = modified[i]
        gold_marker = " ← GOLD" if d == args.did else ""
        was = next((j for j, (dd, _) in enumerate(sae_baseline) if dd == d), -1)
        delta = was - i
        print(f"  {i+1}. did={d}  score={s:.3f}  was_at={was+1} ({delta:+d}){gold_marker}")

    # Optional: write JSON
    out = {
        "qid": args.qid, "did": args.did, "feature": args.feature,
        "mode": args.mode, "top_k": args.top_k,
        "raw_colbert_rank": baseline_rank,
        "sae_baseline_rank": sae_baseline_rank,
        "modified_rank": modified_rank,
        "sae_baseline_score": base_score,
        "modified_score": mod_score,
        "score_delta": mod_score - base_score,
        "rank_delta": modified_rank - sae_baseline_rank,
        "candidates_moved": movers,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
