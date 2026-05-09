"""Batch steering tests: ablate select feature types across our 10
divergent TREC-COVID queries and produce a summary table.

For each query we test 3 features:
  - the always-on backbone (f175)
  - the top backbone-subtracted feature (most query-distinctive)
  - a labeled-COH or labeled-THM feature attributed to MaxSim

Each is tested with --mode ablate, top-k=100. Reports baseline score,
modified score, score delta, gold rank shift, count of candidates that
moved positions.

Usage:
    python -m experiments.steering_battery \\
        --sae-dir <17r dir> \\
        --colbert-cache /data/embeddings/beir/trec-covid-mxbai-edge-32m
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
from experiments.steering_ablation import (  # noqa: E402
    sae_encode_decode_modified, maxsim_score,
)
from experiments.attribute_query_set import DEFAULT_PAIRS  # noqa: E402


BACKBONE_FEATURE = 175  # always-on, mean attribution 0.47


def steering_test(qid: str, did: str, feature_id: int, multiplier: float,
                  q_tokens: np.ndarray, candidate_dids: list[str],
                  dv: np.ndarray, do: np.ndarray, d_ids: list[str],
                  sae: Sae, device: str = "cuda") -> dict:
    base_scores = []
    mod_scores = []
    for cdid in candidate_dids:
        d_tokens = get_doc_tokens(dv, do, d_ids, cdid)
        recon_base, recon_mod = sae_encode_decode_modified(
            d_tokens, sae, feature_id, multiplier, device=device)
        base_scores.append(maxsim_score(q_tokens, recon_base, device=device))
        mod_scores.append(maxsim_score(q_tokens, recon_mod, device=device))
    base_pairs = sorted(zip(candidate_dids, base_scores), key=lambda x: -x[1])
    mod_pairs = sorted(zip(candidate_dids, mod_scores), key=lambda x: -x[1])
    base_rank = next((i for i, (d, _) in enumerate(base_pairs) if d == did), -1)
    mod_rank = next((i for i, (d, _) in enumerate(mod_pairs) if d == did), -1)
    base_score = next(s for d, s in base_pairs if d == did)
    mod_score = next(s for d, s in mod_pairs if d == did)
    base_order = [d for d, _ in base_pairs]
    mod_order = [d for d, _ in mod_pairs]
    movers = sum(1 for i, d in enumerate(mod_order) if base_order[i] != d)
    return {"base_rank": base_rank, "mod_rank": mod_rank,
            "base_score": base_score, "mod_score": mod_score,
            "score_delta": mod_score - base_score,
            "rank_delta": mod_rank - base_rank,
            "candidates_moved": movers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--colbert-cache", required=True)
    ap.add_argument("--colbert-topk-cache",
                    default="/data/embeddings/beir/trec-covid-mxbai-edge-32m/colbert_topk.npz")
    ap.add_argument("--attribution",
                    default="/data/embeddings/beir/trec-covid-attribution.json")
    ap.add_argument("--backbone-subtracted",
                    default="/data/embeddings/beir/trec-covid-attribution-backbone-subtracted.json")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/data/embeddings/beir/trec-covid-steering-battery.json")
    args = ap.parse_args()

    rd = Path(args.sae_dir)
    sae = Sae.load_from_disk(find_sae_ckpt(rd), device=args.device)

    cache = Path(args.colbert_cache)
    qv, qo, q_ids = load_packed(cache, "queries")
    dv, do, d_ids = load_packed(cache, "corpus")

    # ColBERT topk for candidate docs
    npz = np.load(args.colbert_topk_cache, allow_pickle=True)
    topk_mat = npz["top_k"]; cb_q_ids = list(npz["q_ids"]); cb_d_ids = list(npz["d_ids"])

    # Pull labels + backbone-subtracted lists
    labels = json.loads((rd / "feature_labels.json").read_text())["labels"]
    attr_data = json.loads(Path(args.attribution).read_text())
    bb_data = json.loads(Path(args.backbone_subtracted).read_text())
    attr_by_qid = {r["qid"]: r for r in attr_data["records"]}
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

        # Pick the features to test:
        # (a) f175 (backbone)
        # (b) top backbone-subtracted feature
        # (c) top COH/THM feature in attribution
        a_feat = BACKBONE_FEATURE
        bb_top = bb_by_qid.get(qid, {}).get("backbone_subtracted_top", [])
        b_feat = bb_top[0]["feature"] if bb_top else None

        c_feat = None
        for fr in attr_by_qid.get(qid, {}).get("top_all", []):
            j = labels.get(str(fr["feature"]), {}).get("judgment", "")
            if j in ("COHERENT", "THEMATIC"):
                c_feat = fr["feature"]; break

        for label, feat_id in [
            ("backbone(f175)", a_feat),
            (f"bb_distinctive(f{b_feat})", b_feat),
            (f"interp(f{c_feat})", c_feat),
        ]:
            if feat_id is None:
                continue
            res = steering_test(qid, gold_did, feat_id, 0.0,  # ablate
                                q_tok, candidate_dids, dv, do, d_ids,
                                sae, device=args.device)
            row = {"qid": qid, "did": gold_did, "test": label,
                   "feature": feat_id,
                   "judgment": labels.get(str(feat_id), {}).get("judgment", ""),
                   **res}
            rows.append(row)
            jj = row["judgment"][:4] if row["judgment"] else "?"
            print(f"  qid={qid:<3} {label:<24} f{feat_id:<6} [{jj}] "
                  f"score: {res['base_score']:>6.2f} → {res['mod_score']:>6.2f} "
                  f"({res['score_delta']:+.3f})  "
                  f"rank: {res['base_rank']} → {res['mod_rank']}  "
                  f"movers={res['candidates_moved']}")

    Path(args.out).write_text(json.dumps({"rows": rows,
                                            "top_k": args.top_k}, indent=2))
    print(f"\n=== Summary table ===")
    print(f"{'qid':>4} {'test':<28} {'fid':>7} {'judg':<4} "
          f"{'base':>6} {'mod':>6} {'Δ':>7} {'rk_b':>5} {'rk_m':>5} {'mvrs':>5}")
    for r in rows:
        print(f"{r['qid']:>4} {r['test']:<28} {r['feature']:>7} "
              f"{r['judgment'][:4]:<4} "
              f"{r['base_score']:>6.2f} {r['mod_score']:>6.2f} {r['score_delta']:>+7.3f} "
              f"{r['base_rank']:>5} {r['mod_rank']:>5} {r['candidates_moved']:>5}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
