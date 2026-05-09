"""Multi-feature ablation: ablate the top-N backbone-subtracted features
simultaneously for each (qid, did) pair, measure cumulative score and
rank effect.

Single-feature deltas are typically -0.01 to -0.18. Question: do the
top-5 distinctive features cumulatively explain the gold doc's win
margin? If we ablate all 5 at once, does the rank flip consistently?

Usage:
    python -m experiments.steering_multifeat \\
        --sae-dir <SAE dir> \\
        --colbert-cache /data/embeddings/beir/trec-covid-mxbai-edge-32m \\
        --backbone-subtracted /data/embeddings/beir/trec-covid-attribution-<run>-backbone-subtracted.json \\
        --top-n 5
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
    get_query_tokens, get_doc_tokens,
)
from experiments.steering_ablation import maxsim_score  # noqa: E402
from experiments.attribute_query_set import DEFAULT_PAIRS  # noqa: E402


@torch.no_grad()
def sae_modify_multi(d_tokens: np.ndarray, sae: Sae, feature_ids: list[int],
                     multiplier: float, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """SAE forward, then subtract (1 - multiplier) × contribution of EACH
    of the listed features at the largest level. Returns (baseline,
    modified) reconstructions."""
    sae.eval()
    batch = torch.from_numpy(np.ascontiguousarray(d_tokens, dtype=np.float32)).to(device)
    out = sae(batch)
    recon_full = out.sae_out
    acts = out.latent_acts
    idxs = out.latent_indices

    # Sum contributions of all target features
    total_contrib = torch.zeros_like(recon_full)
    for fid in feature_ids:
        mask = (idxs == fid).float()
        f_acts = (acts * mask).sum(dim=1)            # (n,)
        w_row = sae.W_dec[fid]                        # (d_in,)
        total_contrib += f_acts.unsqueeze(-1) * w_row

    recon_modified = recon_full - (1.0 - multiplier) * total_contrib
    return recon_full.cpu().numpy(), recon_modified.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--colbert-cache", required=True)
    ap.add_argument("--colbert-topk-cache",
                    default="/data/embeddings/beir/trec-covid-mxbai-edge-32m/colbert_topk.npz")
    ap.add_argument("--backbone-subtracted", required=True)
    ap.add_argument("--top-n", type=int, default=5,
                    help="number of top backbone-subtracted features to ablate at once")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.sae_dir)
    sae = Sae.load_from_disk(find_sae_ckpt(rd), device=args.device)

    cache = Path(args.colbert_cache)
    qv, qo, q_ids = load_packed(cache, "queries")
    dv, do, d_ids = load_packed(cache, "corpus")

    npz = np.load(args.colbert_topk_cache, allow_pickle=True)
    topk_mat = npz["top_k"]; cb_q_ids = list(npz["q_ids"]); cb_d_ids = list(npz["d_ids"])

    bb_data = json.loads(Path(args.backbone_subtracted).read_text())
    bb_by_qid = {r["qid"]: r for r in bb_data["records"]}

    rows = []
    for qid, gold_did in DEFAULT_PAIRS:
        try:
            qi = cb_q_ids.index(qid)
        except ValueError:
            print(f"SKIP qid={qid}")
            continue
        candidate_dids = [cb_d_ids[j] for j in topk_mat[qi, :args.top_k]]
        if gold_did not in candidate_dids:
            candidate_dids = [gold_did] + candidate_dids
        q_tok = get_query_tokens(qv, qo, q_ids, qid)

        bb_top = bb_by_qid.get(qid, {}).get("backbone_subtracted_top", [])
        feat_ids = [fr["feature"] for fr in bb_top[: args.top_n]]
        if not feat_ids:
            print(f"SKIP qid={qid} (no backbone-subtracted features)")
            continue

        # Run baseline (no ablation) and full ablation
        base_scores = []
        mod_scores = []
        for cdid in candidate_dids:
            d_tokens = get_doc_tokens(dv, do, d_ids, cdid)
            recon_base, recon_mod = sae_modify_multi(
                d_tokens, sae, feat_ids, 0.0, device=args.device)
            base_scores.append(maxsim_score(q_tok, recon_base, device=args.device))
            mod_scores.append(maxsim_score(q_tok, recon_mod, device=args.device))
        base_pairs = sorted(zip(candidate_dids, base_scores), key=lambda x: -x[1])
        mod_pairs = sorted(zip(candidate_dids, mod_scores), key=lambda x: -x[1])
        base_rank = next((i for i, (d, _) in enumerate(base_pairs) if d == gold_did), -1)
        mod_rank = next((i for i, (d, _) in enumerate(mod_pairs) if d == gold_did), -1)
        base_score = next(s for d, s in base_pairs if d == gold_did)
        mod_score = next(s for d, s in mod_pairs if d == gold_did)
        movers = sum(1 for i, (d, _) in enumerate(mod_pairs) if base_pairs[i][0] != d)

        row = {"qid": qid, "did": gold_did,
               "feature_ids": feat_ids,
               "base_score": base_score, "mod_score": mod_score,
               "score_delta": mod_score - base_score,
               "base_rank": base_rank, "mod_rank": mod_rank,
               "rank_delta": mod_rank - base_rank,
               "candidates_moved": movers}
        rows.append(row)
        print(f"  qid={qid:<3}  ablate {len(feat_ids)} features {feat_ids}  "
              f"score: {base_score:>6.2f} → {mod_score:>6.2f} "
              f"({mod_score - base_score:+.3f})  "
              f"rank: {base_rank} → {mod_rank} ({mod_rank - base_rank:+d})  "
              f"movers={movers}")

    out_path = (Path(args.out) if args.out
                else Path(args.backbone_subtracted).parent /
                Path(args.backbone_subtracted).stem.replace(
                    "-backbone-subtracted", f"-multifeat-top{args.top_n}.json").lstrip("/"))
    out_path.write_text(json.dumps({
        "sae_dir": str(rd), "top_n": args.top_n, "top_k": args.top_k,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {out_path}")
    print(f"\n=== Summary: top-{args.top_n} ablation effect ===")
    print(f"{'qid':>4} {'rank_b':>6} {'rank_m':>6} {'Δ':>5} {'score_Δ':>8}  {'features':<30}")
    for r in rows:
        feats = ",".join(f"f{f}" for f in r["feature_ids"])
        print(f"{r['qid']:>4} {r['base_rank']:>6} {r['mod_rank']:>6} "
              f"{r['rank_delta']:>+5d} {r['score_delta']:>+8.3f}  {feats[:30]}")


if __name__ == "__main__":
    main()
