"""
Autoresearch evaluation — LOCKED FILE, do not edit.

Fast eval suite for 5-minute SAE experiment loops:
  1. CLINC150 (150-class intent classification) — fine-grained probe accuracy
  2. SciFact (retrieval nDCG@10) — information preservation
  3. MMCS (decoder redundancy) — feature quality
  4. FVU — reconstruction quality

Composite score: CLINC150*0.4 + SciFact*0.3 + (1-MMCS)*0.3 (higher = better)

Caches embeddings so re-evaluation only costs SAE encode + probes (~25s).
"""

import json
import os
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _get_or_embed(name, texts, model_name, normalize=True):
    """Cache embeddings to disk so we only embed once per model."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"{name}_{safe_model}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embs = model.encode(list(texts), batch_size=256, show_progress_bar=False, normalize_embeddings=normalize)
    np.save(cache_path, embs)
    return embs


def load_clinc150():
    from datasets import load_dataset
    ds = load_dataset("clinc_oos", "plus", split="test")
    return list(ds["text"]), np.array(ds["intent"])


def load_scifact():
    from datasets import load_dataset
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    default_ds = load_dataset("mteb/scifact", "default", split="test")

    queries = {row["_id"]: row["text"] for row in queries_ds}
    corpus = {row["_id"]: (row.get("title", "") + " " + row["text"]).strip() for row in corpus_ds}
    qrels = {}
    for row in default_ds:
        qid, cid, score = row["query-id"], row["corpus-id"], row["score"]
        qrels.setdefault(qid, {})[cid] = score

    return queries, corpus, qrels


def sae_encode(sae, embeddings, batch_size=512, device="cpu"):
    sae.eval()
    dev = torch.device(device)
    sae = sae.to(dev)
    all_recon, all_sparse = [], []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(dev)
            enc = sae.encode(batch)
            recon = sae.decode(enc.top_acts, enc.top_indices)
            all_recon.append(recon.cpu().numpy())
            sparse = torch.zeros(len(batch), sae.num_latents, device=dev)
            sparse.scatter_(1, enc.top_indices, enc.top_acts)
            all_sparse.append(sparse.cpu().numpy())
    return np.concatenate(all_recon), np.concatenate(all_sparse)


def eval_clinc150(sae, model_name, device="cpu"):
    """CLINC150 probe accuracy (sparse features vs raw)."""
    texts, labels = load_clinc150()
    embs = _get_or_embed("clinc150", texts, model_name)
    _, sparse = sae_encode(sae, embs, device=device)

    n = len(embs)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]

    # Raw baseline
    clf_raw = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
    clf_raw.fit(embs[tr], labels[tr])
    raw_acc = accuracy_score(labels[te], clf_raw.predict(embs[te]))

    # Sparse features
    clf_sparse = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
    clf_sparse.fit(sparse[tr], labels[tr])
    sparse_acc = accuracy_score(labels[te], clf_sparse.predict(sparse[te]))

    return {"raw": raw_acc, "sparse": sparse_acc, "gap": raw_acc - sparse_acc}


def eval_scifact(sae, model_name, device="cpu"):
    """SciFact retrieval nDCG@10 (sparse features vs raw)."""
    queries, corpus, qrels = load_scifact()
    q_ids = list(queries.keys())
    c_ids = list(corpus.keys())

    q_embs = _get_or_embed("scifact_queries", list(queries.values()), model_name)
    c_embs = _get_or_embed("scifact_corpus", list(corpus.values()), model_name)

    def ndcg10(q_emb, c_emb):
        sims = cosine_similarity(q_emb, c_emb)
        scores = []
        for i, qid in enumerate(q_ids):
            if qid not in qrels:
                continue
            ranked = np.argsort(-sims[i])[:10]
            dcg = sum(qrels[qid].get(c_ids[j], 0) / np.log2(r + 2) for r, j in enumerate(ranked))
            ideal = sorted(qrels[qid].values(), reverse=True)[:10]
            idcg = sum(rel / np.log2(r + 2) for r, rel in enumerate(ideal))
            if idcg > 0:
                scores.append(dcg / idcg)
        return np.mean(scores) if scores else 0.0

    raw_ndcg = ndcg10(q_embs, c_embs)

    _, q_sparse = sae_encode(sae, q_embs, device=device)
    _, c_sparse = sae_encode(sae, c_embs, device=device)
    sparse_ndcg = ndcg10(q_sparse, c_sparse)

    return {"raw": raw_ndcg, "sparse": sparse_ndcg, "gap": raw_ndcg - sparse_ndcg}


def eval_feature_quality(sae, model_name, device="cpu"):
    """MMCS + utilization from decoder weights and CLINC150 activations."""
    W_dec = sae.W_dec.data.cpu().float()
    W_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)
    cos = W_norm @ W_norm.T
    cos.fill_diagonal_(0)
    mmcs = cos.max(dim=1).values.mean().item()

    texts, _ = load_clinc150()
    embs = _get_or_embed("clinc150", texts, model_name)
    fire_count = torch.zeros(sae.num_latents)
    sae.eval()
    dev = torch.device(device)
    sae = sae.to(dev)
    with torch.no_grad():
        for i in range(0, len(embs), 512):
            batch = torch.tensor(embs[i:i+512], dtype=torch.float32).to(dev)
            enc = sae.encode(batch)
            for idx in enc.top_indices.cpu():
                fire_count[idx] += 1

    active = int((fire_count > 0).sum().item())
    return {"mmcs": mmcs, "active_features": active, "total_features": sae.num_latents,
            "utilization": active / sae.num_latents}


def eval_fvu(sae, model_name, device="cpu"):
    """FVU on CLINC150 embeddings."""
    texts, _ = load_clinc150()
    embs = _get_or_embed("clinc150", texts, model_name)
    recon, _ = sae_encode(sae, embs, device=device)
    mse = ((embs - recon) ** 2).sum()
    var = ((embs - embs.mean(0)) ** 2).sum()
    return {"fvu": float(mse / var)}


def evaluate(sae, model_name, device="cpu"):
    """Run full fast eval suite. Returns dict with all metrics + composite score."""
    t0 = time.time()

    clinc = eval_clinc150(sae, model_name, device)
    scifact = eval_scifact(sae, model_name, device)
    fq = eval_feature_quality(sae, model_name, device)
    fvu = eval_fvu(sae, model_name, device)

    composite = clinc["sparse"] * 0.4 + scifact["sparse"] * 0.3 + (1 - fq["mmcs"]) * 0.3
    eval_time = time.time() - t0

    return {
        "composite_score": composite,
        "clinc150_sparse": clinc["sparse"],
        "clinc150_raw": clinc["raw"],
        "clinc150_gap": clinc["gap"],
        "scifact_sparse": scifact["sparse"],
        "scifact_raw": scifact["raw"],
        "scifact_gap": scifact["gap"],
        "mmcs": fq["mmcs"],
        "active_features": fq["active_features"],
        "total_features": fq["total_features"],
        "fvu": fvu["fvu"],
        "eval_time_s": eval_time,
    }


def format_results(results):
    """Format results for grep-friendly output."""
    lines = ["---"]
    for k, v in results.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.6f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
