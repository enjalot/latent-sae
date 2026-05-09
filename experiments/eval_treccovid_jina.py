"""TREC-COVID retrieval eval for jina pooled SAEs.

Mirrors the SciFact retrieval block in eval_probes.py but on TREC-COVID
(BEIR's mteb/trec-covid). Allows direct comparison with the existing
ColBERT 17b/r/s TREC-COVID results.

Usage:
    python -m experiments.eval_treccovid_jina \\
        --sae-path .../checkpoints/sae_matryoshka_X.pooled \\
        --output .../eval_treccovid.json
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch


def load_treccovid():
    from datasets import load_dataset
    queries_ds = load_dataset("mteb/trec-covid", "queries", split="queries")
    corpus_ds = load_dataset("mteb/trec-covid", "corpus", split="corpus")
    qrels_ds = load_dataset("mteb/trec-covid", "default", split="test")
    queries = {row["_id"]: row["text"] for row in queries_ds}
    corpus = {row["_id"]: ((row.get("title", "") or "") + " " + (row["text"] or "")).strip()
              for row in corpus_ds}
    qrels = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {qid: txt for qid, txt in queries.items() if qid in qrels}
    return queries, corpus, qrels


def ndcg_at_k(qrels: dict, ranked: dict, k: int = 10) -> float:
    """nDCG@k averaged over queries."""
    scores = []
    for qid, gold in qrels.items():
        if qid not in ranked: continue
        # DCG: sum (rel / log2(i+2)) for top-k
        top = ranked[qid][:k]
        dcg = sum(gold.get(did, 0) / np.log2(i + 2) for i, did in enumerate(top))
        # ideal: sort gold by relevance desc, take top-k
        ideal = sorted(gold.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        scores.append(dcg / idcg if idcg > 0 else 0)
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-path", required=True)
    ap.add_argument("--embedding-model", default="jinaai/jina-embeddings-v5-text-nano-retrieval")
    ap.add_argument("--corpus-cap", type=int, default=20000,
                    help="cap corpus size to fit in GPU memory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from latentsae.sae import Sae
    from sentence_transformers import SentenceTransformer

    queries, corpus, qrels = load_treccovid()
    print(f"TREC-COVID: {len(queries)} queries, {len(corpus)} corpus docs")

    # Subsample corpus if large (TREC-COVID full corpus is 171k docs).
    # Keep ALL positives, then fill with random negatives up to corpus_cap.
    if len(corpus) > args.corpus_cap:
        positives = set()
        for q in qrels.values():
            positives.update(q.keys())
        positives &= set(corpus.keys())
        rng = np.random.default_rng(42)
        remaining = list(set(corpus.keys()) - positives)
        rng.shuffle(remaining)
        # Always keep positives; add negatives up to cap (don't go below positives)
        n_neg = max(0, args.corpus_cap - len(positives))
        keep = positives | set(remaining[:n_neg])
        corpus = {k: v for k, v in corpus.items() if k in keep}
        print(f"  subsampled corpus to {len(corpus)} (kept all {len(positives)} positives + {n_neg} negatives)")

    qids = list(queries); cids = list(corpus)
    q_texts = [queries[i] for i in qids]
    c_texts = [corpus[i] for i in cids]

    sae = Sae.load_from_disk(args.sae_path, device=args.device)
    print(f"SAE: {sae.num_latents} latents, d_in={sae.d_in}")

    # Encode with embedder; SAE on CPU temporarily to free GPU for attention
    sae.cpu(); gc.collect(); torch.cuda.empty_cache()
    emb_model = SentenceTransformer(args.embedding_model, trust_remote_code=True,
                                     device=args.device)
    # TREC-COVID corpus docs are full medical abstracts; cap seq length so
    # attention mask doesn't balloon to many GB at higher batch sizes
    emb_model.max_seq_length = 256
    print(f"encoding {len(q_texts)} queries + {len(c_texts)} corpus docs (max_seq_length=256)...")
    q_emb = emb_model.encode(q_texts, batch_size=32, show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    c_emb = emb_model.encode(c_texts, batch_size=32, show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    del emb_model
    gc.collect(); torch.cuda.empty_cache()
    sae.to(args.device)

    print("SAE forward pass...")
    with torch.no_grad():
        def sae_pass(embs, batch=2048):
            recons, sparses = [], []
            for s in range(0, len(embs), batch):
                e = min(s + batch, len(embs))
                x = torch.from_numpy(embs[s:e]).to(args.device)
                out = sae(x)
                recons.append(out.sae_out.cpu().numpy())
                top_acts = out.latent_acts.cpu().numpy()
                top_idx = out.latent_indices.cpu().numpy()
                sp = np.zeros((e - s, sae.num_latents), dtype=np.float32)
                for j in range(e - s):
                    sp[j, top_idx[j]] = top_acts[j]
                sparses.append(sp)
            return np.concatenate(recons), np.concatenate(sparses)

        q_recon, q_sparse = sae_pass(q_emb)
        c_recon, c_sparse = sae_pass(c_emb)

    # Compute retrieval per kind
    def rank(qe, ce):
        # cosine similarity, queries assumed normalized; renormalize anyway
        qn = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-12)
        cn = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-12)
        sims = qn @ cn.T  # (Q, C)
        idx = np.argsort(-sims, axis=1)
        return {qids[i]: [cids[j] for j in idx[i, :100]] for i in range(len(qids))}

    print("computing nDCG@10...")
    results = {}
    for kind, qe, ce in [("raw", q_emb, c_emb),
                          ("recon", q_recon, c_recon),
                          ("sparse", q_sparse, c_sparse)]:
        ranked = rank(qe, ce)
        ndcg = ndcg_at_k(qrels, ranked, k=10)
        results[kind] = {"ndcg@10": ndcg, "n_queries": len(qids), "n_corpus": len(cids)}
        print(f"  {kind:>6}: nDCG@10 = {ndcg:.4f}")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
