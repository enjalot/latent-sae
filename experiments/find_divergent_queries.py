"""Find TREC-COVID queries where ColBERT MaxSim outranks pooled embedders
on the gold-relevant document.

Computes for each query the rank that each retriever assigns to the
*highest-graded* relevant doc (qrels score 2 = highly relevant). A
query is "divergent" if ColBERT ranks the gold doc top-K but a pooled
embedder ranks it far worse — those are the queries where late-
interaction is doing the work.

Inputs:
  - cached top-k indices from each retriever (ColBERT + MiniLM + Jina v5)
  - BEIR qrels for ground truth

Output:
  - JSON with per-query ranks across retrievers + a divergence score
  - Sorted list of "ColBERT-wins-big" queries with the gold doc id

Usage:
    python -m experiments.find_divergent_queries \\
        --dataset trec-covid \\
        --colbert-results /data/embeddings/beir/colbert_retrieval_topk.json \\
        --pooled-cache /data/embeddings/beir/trec-covid-pooled-minilm \\
        --pooled-cache /data/embeddings/beir/trec-covid-pooled-jina-v5-small
"""
import argparse
import json
from pathlib import Path

import numpy as np


BEIR_PATHS = {
    "trec-covid": "mteb/trec-covid",
    "scifact": "mteb/scifact",
}


def load_qrels(dataset: str) -> dict[str, dict[str, int]]:
    from datasets import load_dataset
    qrels_ds = load_dataset(BEIR_PATHS[dataset], "default", split="test")
    out: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        out.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    return out


def rank_of_gold(top_k_indices: np.ndarray, q_ids: list[str], d_ids: list[str],
                 qrels: dict, min_score: int = 2) -> dict[str, dict]:
    """For each query, return the best rank (smallest) among gold docs."""
    out: dict[str, dict] = {}
    d_idx = {d: i for i, d in enumerate(d_ids)}
    for i, qid in enumerate(q_ids):
        relevant = qrels.get(qid, {})
        gold = [d for d, s in relevant.items() if s >= min_score]
        if not gold:
            out[qid] = {"best_gold_rank": None, "n_gold": 0, "best_gold_did": None}
            continue
        ranked = top_k_indices[i]
        best_rank = None
        best_did = None
        for rank, j in enumerate(ranked):
            cid = d_ids[j]
            if cid in gold:
                if best_rank is None:
                    best_rank = rank
                    best_did = cid
                    break
        out[qid] = {"best_gold_rank": best_rank, "n_gold": len(gold),
                    "best_gold_did": best_did}
    return out


def colbert_topk_from_eval(colbert_eval_results_path: Path,
                            q_ids: list[str], d_ids: list[str]) -> np.ndarray:
    """Reconstruct a top-k indices matrix from a saved ColBERT retrieval JSON.

    Falls back: if the saved JSON only has aggregate metrics, re-rank by
    running a small re-derivation against the cached MaxSim scores. We
    don't have that path implemented here; expect the eval to have been
    run with a per-query score matrix saved.
    """
    raise NotImplementedError(
        "ColBERT topk reconstruction not yet wired up — pass --colbert-topk-cache "
        "with a precomputed (n_q, k) array, or run --recompute-colbert.")


def recompute_colbert_topk(dataset: str, k: int = 100,
                            cache_root: str = "/data/embeddings/beir") -> tuple[np.ndarray, list[str], list[str]]:
    """Run dense MaxSim against the cached ColBERT vectors and return top-k.

    Imports from eval_colbert_retrieval to reuse maxsim code.
    """
    import sys
    sys.path.insert(0, "/home/enjalot/code/latent-sae")
    from experiments.eval_colbert_retrieval import (  # type: ignore
        load_beir, embed_or_load, maxsim_scores, MODEL_SLUG)
    queries, corpus, qrels = load_beir(dataset)
    q_ids = sorted(queries.keys())
    d_ids = sorted(corpus.keys())
    cache_dir = Path(cache_root) / f"{dataset}-{MODEL_SLUG}"
    qv, qo, _ = embed_or_load("queries", q_ids, [queries[q] for q in q_ids],
                              cache_dir, is_query=True)
    dv, do, _ = embed_or_load("corpus", d_ids, [corpus[d] for d in d_ids],
                              cache_dir, is_query=False)
    print("computing dense MaxSim ...")
    scores = maxsim_scores(qv, qo, dv, do, device="cuda")  # (n_q, n_d)
    top_k = np.argsort(-scores, axis=1)[:, :k]
    return top_k, q_ids, d_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--pooled-cache", action="append", required=True,
                    help="path to pooled cache dir (with topk_indices.npy + topk_meta.json). May repeat.")
    ap.add_argument("--colbert-topk-cache", default=None,
                    help="precomputed ColBERT topk .npz (with arrays top_k, q_ids, d_ids)")
    ap.add_argument("--recompute-colbert", action="store_true",
                    help="re-compute ColBERT MaxSim on the fly using cached vectors")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-divergent", type=int, default=20)
    args = ap.parse_args()

    qrels = load_qrels(args.dataset)

    # ColBERT
    if args.recompute_colbert:
        cb_top, q_ids, d_ids = recompute_colbert_topk(args.dataset)
        out_cache = Path(f"/data/embeddings/beir/{args.dataset}-mxbai-edge-32m")
        np.savez(out_cache / "colbert_topk.npz",
                 top_k=cb_top, q_ids=q_ids, d_ids=d_ids)
        print(f"saved ColBERT topk to {out_cache / 'colbert_topk.npz'}")
    elif args.colbert_topk_cache:
        d = np.load(args.colbert_topk_cache, allow_pickle=True)
        cb_top = d["top_k"]; q_ids = list(d["q_ids"]); d_ids = list(d["d_ids"])
    else:
        # Try the conventional location
        p = Path(f"/data/embeddings/beir/{args.dataset}-mxbai-edge-32m/colbert_topk.npz")
        if p.exists():
            d = np.load(p, allow_pickle=True)
            cb_top = d["top_k"]; q_ids = list(d["q_ids"]); d_ids = list(d["d_ids"])
        else:
            raise SystemExit("Need either --colbert-topk-cache or --recompute-colbert "
                             "(no cached file found at " + str(p) + ")")

    cb_ranks = rank_of_gold(cb_top, q_ids, d_ids, qrels)

    # Pooled retrievers
    pooled_ranks_per_model: dict[str, dict[str, dict]] = {}
    for cache_dir in args.pooled_cache:
        cd = Path(cache_dir)
        meta = json.loads((cd / "topk_meta.json").read_text())
        slug = meta.get("slug", cd.name)
        top = np.load(cd / "topk_indices.npy")
        # Use ids from meta, which were sorted in the eval
        p_q_ids = meta["q_ids"]; p_d_ids = meta["d_ids"]
        if p_q_ids != q_ids or p_d_ids != d_ids:
            print(f"WARN: id ordering differs for {slug} — re-aligning")
        pooled_ranks_per_model[slug] = rank_of_gold(top, p_q_ids, p_d_ids, qrels)

    # Build per-query summary
    rows = []
    for qid in q_ids:
        cb = cb_ranks[qid]
        if cb["best_gold_rank"] is None:
            continue
        row = {
            "qid": qid,
            "best_gold_did": cb["best_gold_did"],
            "n_gold": cb["n_gold"],
            "colbert_rank": cb["best_gold_rank"],
        }
        worst_pooled_rank = -1
        for slug, ranks in pooled_ranks_per_model.items():
            r = ranks[qid]["best_gold_rank"]
            row[f"{slug}_rank"] = r if r is not None else -1
            if r is None:
                worst_pooled_rank = max(worst_pooled_rank, 999)
            else:
                worst_pooled_rank = max(worst_pooled_rank, r)
        # Divergence: worst pooled rank minus ColBERT rank (positive = ColBERT wins)
        row["divergence"] = worst_pooled_rank - cb["best_gold_rank"]
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: -r["divergence"])

    print(f"\n=== Top {args.top_divergent} queries where ColBERT wins biggest over pooled ===")
    print(f"{'qid':<6} {'gold_did':<14} {'cb_rank':>7} ", end="")
    pooled_slugs = list(pooled_ranks_per_model.keys())
    for slug in pooled_slugs:
        print(f"{slug + '_rank':>15}", end=" ")
    print(" diverg")
    for r in rows_sorted[:args.top_divergent]:
        print(f"{r['qid']:<6} {r['best_gold_did']:<14} {r['colbert_rank']:>7} ", end="")
        for slug in pooled_slugs:
            v = r[f"{slug}_rank"]
            print(f"{v:>15}", end=" ")
        print(f" {r['divergence']:>6}")

    out_path = (Path(args.out) if args.out
                else Path(f"/data/embeddings/beir/{args.dataset}-divergence.json"))
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "pooled_models": list(pooled_ranks_per_model.keys()),
        "rows": rows_sorted,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
