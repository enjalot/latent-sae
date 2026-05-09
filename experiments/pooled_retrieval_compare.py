"""Embed BEIR corpus + queries with pooled embedders and compare retrieval
ranking against cached ColBERT MaxSim.

Outputs a JSON with per-query top-k for each system and a divergence
score. Used to pick queries where ColBERT MaxSim wins big over pooled
single-vector retrieval — those are our targets for MaxSim attribution.

Cache layout (per dataset, per model):
    /data/embeddings/beir/<dataset>-pooled-<slug>/
      corpus_vectors.npy   (n_docs, dim)
      query_vectors.npy    (n_queries, dim)
      ids.json              {corpus_ids, query_ids}

Usage:
    python -m experiments.pooled_retrieval_compare \\
        --dataset trec-covid \\
        --model jinaai/jina-embeddings-v5-text-small-retrieval --slug jina-v5-small
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


def load_beir(dataset_key: str):
    """Load queries, corpus, qrels for a BEIR dataset (TREC-COVID, scifact, etc).

    Mirrors eval_colbert_retrieval.load_beir but stays self-contained so this
    file doesn't import the colbert eval.
    """
    from datasets import load_dataset
    paths = {
        "trec-covid": "mteb/trec-covid",
        "scifact": "mteb/scifact",
    }
    cfg = paths[dataset_key]
    queries_ds = load_dataset(cfg, "queries", split="queries")
    corpus_ds = load_dataset(cfg, "corpus", split="corpus")
    qrels_ds = load_dataset(cfg, "default", split="test")

    queries = {row["_id"]: row["text"] for row in queries_ds}
    corpus = {row["_id"]: (row.get("title", "") + " " + row["text"]).strip()
              for row in corpus_ds}
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {qid: txt for qid, txt in queries.items() if qid in qrels}
    return queries, corpus, qrels


def embed_or_load(model_id: str, slug: str, dataset: str,
                  ids: list[str], texts: list[str], kind: str,
                  cache_root: str = "/data/embeddings/beir",
                  device: str = "cuda") -> tuple[np.ndarray, list[str]]:
    cache = Path(cache_root) / f"{dataset}-pooled-{slug}"
    cache.mkdir(parents=True, exist_ok=True)
    vecs_path = cache / f"{kind}_vectors.npy"
    ids_path = cache / f"{kind}_ids.json"
    if vecs_path.exists() and ids_path.exists():
        cached_ids = json.loads(ids_path.read_text())
        if cached_ids == ids:
            print(f"  [{slug}/{kind}] cache hit — {vecs_path}")
            return np.load(vecs_path), cached_ids
        print(f"  [{slug}/{kind}] cache mismatch (different ids), re-embedding")

    from sentence_transformers import SentenceTransformer
    print(f"  [{slug}/{kind}] loading {model_id} ...")
    model = SentenceTransformer(model_id, cache_folder="/data/hf/sentence-transformers",
                                 device=device, trust_remote_code=True)
    print(f"  [{slug}/{kind}] encoding {len(texts)} {kind} ...")
    t = time.monotonic()
    vecs = model.encode(texts, batch_size=128, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True)
    print(f"  [{slug}/{kind}] {vecs.shape} in {time.monotonic()-t:.1f}s")
    np.save(vecs_path, vecs.astype(np.float32))
    ids_path.write_text(json.dumps(ids))
    return vecs, ids


def cosine_topk(q_vecs: np.ndarray, d_vecs: np.ndarray, k: int = 100) -> np.ndarray:
    """Return (n_queries, k) doc-index matrix sorted by cosine descending.

    Vectors are assumed l2-normalized (they are when normalize_embeddings=True).
    """
    sims = q_vecs @ d_vecs.T  # (n_q, n_d)
    return np.argsort(-sims, axis=1)[:, :k]


def ndcg_at_k(top_k: np.ndarray, q_ids: list[str], d_ids: list[str],
              qrels: dict, k: int = 10) -> dict:
    """nDCG@k for a (n_queries, k) sorted-doc-index matrix."""
    ndcg_per = []
    for i, qid in enumerate(q_ids):
        relevant = qrels.get(qid, {})
        if not relevant:
            continue
        dcg = 0.0
        for rank, j in enumerate(top_k[i, :k]):
            cid = d_ids[j]
            g = relevant.get(cid, 0)
            if g > 0:
                dcg += (2 ** g - 1) / np.log2(rank + 2)
        ideal_grades = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / np.log2(r + 2)
                   for r, g in enumerate(ideal_grades))
        ndcg_per.append(dcg / idcg if idcg > 0 else 0.0)
    return {f"ndcg@{k}": float(np.mean(ndcg_per)) if ndcg_per else 0.0,
            "n_queries": len(ndcg_per)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--model", required=True,
                    help="HuggingFace ID for sentence-transformers model")
    ap.add_argument("--slug", required=True,
                    help="short cache slug, e.g. minilm or jina-v5-small")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"=== {args.dataset} × {args.model} ===")
    queries, corpus, qrels = load_beir(args.dataset)
    q_ids = sorted(queries.keys())
    d_ids = sorted(corpus.keys())
    q_texts = [queries[q] for q in q_ids]
    d_texts = [corpus[d] for d in d_ids]
    print(f"{len(q_ids)} queries, {len(d_ids)} docs")

    q_vecs, _ = embed_or_load(args.model, args.slug, args.dataset, q_ids, q_texts,
                               "queries", device=args.device)
    d_vecs, _ = embed_or_load(args.model, args.slug, args.dataset, d_ids, d_texts,
                               "corpus", device=args.device)

    print(f"\ncosine retrieval, top-{args.top_k} ...")
    top_k = cosine_topk(q_vecs, d_vecs, k=args.top_k)
    metrics = ndcg_at_k(top_k, q_ids, d_ids, qrels, k=10)
    print(f"  {args.slug}: nDCG@10 = {metrics['ndcg@10']:.4f}  ({metrics['n_queries']} queries)")

    out = Path(f"/data/embeddings/beir/{args.dataset}-pooled-{args.slug}")
    np.save(out / "topk_indices.npy", top_k)
    (out / "topk_meta.json").write_text(json.dumps({
        "model": args.model, "slug": args.slug, "dataset": args.dataset,
        "top_k": args.top_k, **metrics,
        "q_ids": q_ids, "d_ids": d_ids,
    }))
    print(f"wrote {out / 'topk_indices.npy'} and topk_meta.json")


if __name__ == "__main__":
    main()
