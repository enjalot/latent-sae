"""
Evaluate SAE quality via downstream probe tasks and feature quality metrics.

Probes (logistic regression on raw vs reconstructed vs sparse activations):
  - AG News (4-class topic), SST-2 (2-class sentiment), STS-B (similarity)
  - BANKING77 (77 fine-grained intents), CLINC150 (150 intents)

Retrieval: SciFact (nDCG@10 on cosine similarity)
k-Sparse probing: accuracy with only top-1, top-5, top-10, top-20 features
Feature quality: decoder redundancy (MMCS), utilization, activation entropy

Usage:
  python -m experiments.eval_probes --sae-path checkpoints/sae_topk_64_8.xxx
  python -m experiments.eval_probes --sae-path checkpoints/sae_topk_64_8.xxx --suite hard
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr


# ── Data Loaders ──

def load_ag_news(max_samples=5000):
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="test")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds["text"], ds["label"], "AG News (4-class)"


def load_sst2(max_samples=5000):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds["sentence"], ds["label"], "SST-2 (2-class)"


def load_stsb(max_samples=5000):
    from datasets import load_dataset
    try:
        ds = load_dataset("glue", "stsb", split="validation")
    except (ValueError, FileNotFoundError):
        ds = load_dataset("nyu-mll/glue", "stsb", split="validation")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return (ds["sentence1"], ds["sentence2"]), ds["label"], "STS-B (similarity)"


def load_banking77(max_samples=5000):
    from datasets import load_dataset
    try:
        ds = load_dataset("banking77", split="test")
    except Exception:
        ds = load_dataset("PolyAI/banking77", split="test")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds["text"], ds["label"], "BANKING77 (77-class)"


def load_clinc150(max_samples=5000):
    from datasets import load_dataset
    try:
        ds = load_dataset("clinc_oos", "plus", split="test")
        return ds["text"], ds["intent"], "CLINC150 (150-class)"
    except Exception:
        ds = load_dataset("clinc/clinc150", "plus", split="test")
        return ds["text"], ds["intent"], "CLINC150 (150-class)"


def load_scifact():
    """Load SciFact retrieval dataset. Returns queries, corpus, qrels."""
    from datasets import load_dataset
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    # Load qrels (relevance judgments)
    default_ds = load_dataset("mteb/scifact", "default", split="test")

    queries = {row["_id"]: row["text"] for row in queries_ds}
    corpus = {row["_id"]: (row.get("title", "") + " " + row["text"]).strip() for row in corpus_ds}

    # Parse qrels from the default split
    qrels = {}
    for row in default_ds:
        qid = row["query-id"]
        cid = row["corpus-id"]
        score = row["score"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][cid] = score

    return queries, corpus, qrels


# ── Embedding & SAE Encoding ──

def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    if isinstance(texts, tuple):
        emb1 = model.encode(texts[0], batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        emb2 = model.encode(texts[1], batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        return emb1, emb2
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)


def sae_encode(sae, embeddings, batch_size=512, device="cpu"):
    """Encode through SAE. Returns reconstructed (dense) and sparse activations.
    Sparse is returned as dense array but built in small batches to limit memory."""
    sae.eval()
    dev = torch.device(device)
    sae = sae.to(dev)

    all_reconstructed = []
    all_sparse = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(dev)
            enc = sae.encode(batch)
            recon = sae.decode(enc.top_acts, enc.top_indices)
            all_reconstructed.append(recon.cpu().numpy())
            # Build sparse on CPU immediately to free GPU/MPS memory
            sparse = np.zeros((len(batch), sae.num_latents), dtype=np.float32)
            idx = enc.top_indices.cpu().numpy()
            acts = enc.top_acts.cpu().numpy()
            for j in range(len(batch)):
                sparse[j, idx[j]] = acts[j]
            all_sparse.append(sparse)

    return np.concatenate(all_reconstructed), np.concatenate(all_sparse)


def sae_encode_topk_sparse(sae, embeddings, k, batch_size=512, device="cpu"):
    """Encode and return sparse activations with only top-k features (for k-sparse probing)."""
    sae.eval()
    dev = torch.device(device)
    sae = sae.to(dev)

    all_sparse = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(dev)
            enc = sae.encode(batch)
            # Full sparse, then keep only top-k by activation magnitude
            sparse = torch.zeros(len(batch), sae.num_latents, device=dev)
            sparse.scatter_(1, enc.top_indices, enc.top_acts)
            if k < enc.top_acts.shape[-1]:
                topk_vals, topk_idx = sparse.abs().topk(k, dim=-1)
                mask = torch.zeros_like(sparse)
                mask.scatter_(1, topk_idx, 1.0)
                sparse = sparse * mask
            all_sparse.append(sparse.cpu().numpy())

    return np.concatenate(all_sparse)


# ── Evaluation Functions ──

def probe_classification(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }


def probe_similarity(emb1_test, emb2_test, labels_test):
    from sklearn.metrics.pairwise import cosine_similarity
    sims = np.array([cosine_similarity(emb1_test[i:i+1], emb2_test[i:i+1])[0, 0]
                     for i in range(len(emb1_test))])
    corr, _ = spearmanr(sims, labels_test)
    return {"spearman": corr}


def eval_retrieval(query_embs, corpus_embs, query_ids, corpus_ids, qrels, k=10):
    """Compute nDCG@k for retrieval. Memory-efficient: one query at a time."""
    ndcg_scores = []
    # Normalize corpus once for dot-product = cosine similarity
    corpus_norms = np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    corpus_normed = corpus_embs / (corpus_norms + 1e-8)

    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        # Compute similarities for this query only (1 x corpus_size)
        q = query_embs[i:i+1]
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        sims = (q_norm @ corpus_normed.T).ravel()

        ranked_idx = np.argpartition(-sims, k)[:k]
        ranked_idx = ranked_idx[np.argsort(-sims[ranked_idx])]
        ranked_cids = [corpus_ids[j] for j in ranked_idx]

        dcg = sum(qrels[qid].get(cid, 0) / np.log2(rank + 2) for rank, cid in enumerate(ranked_cids))
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum(r / np.log2(rank + 2) for rank, r in enumerate(ideal_rels))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return {"ndcg@10": np.mean(ndcg_scores) if ndcg_scores else 0.0, "n_queries": len(ndcg_scores)}


def eval_feature_quality(sae, embeddings, batch_size=512, device="cpu"):
    """Compute feature quality metrics from SAE weights and activations."""
    sae.eval()
    dev = torch.device(device)
    sae = sae.to(dev)

    # 1. Decoder redundancy (MMCS - Mean Max Cosine Similarity)
    W_dec = sae.W_dec.data.to("cpu").float()
    W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = W_dec_norm @ W_dec_norm.T
    cos_sim.fill_diagonal_(0)  # exclude self-similarity
    mmcs = cos_sim.max(dim=1).values.mean().item()

    # 2. Feature utilization — what fraction of features fire on this corpus
    fire_count = torch.zeros(sae.num_latents)
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(dev)
            enc = sae.encode(batch)
            for idx in enc.top_indices.cpu():
                fire_count[idx] += 1
            total_samples += len(batch)

    feature_fire_rate = fire_count / total_samples
    utilization = (fire_count > 0).float().mean().item()
    dead_features = (fire_count == 0).float().mean().item()

    # 3. Activation entropy per feature (specificity)
    # Features that fire on everything have high entropy (bad)
    # Features that fire on specific inputs have low entropy (good)
    probs = fire_count / (fire_count.sum() + 1e-8)
    probs = probs[probs > 0]
    activation_entropy = -(probs * probs.log()).sum().item()
    max_entropy = np.log(sae.num_latents)
    normalized_entropy = activation_entropy / max_entropy

    # 4. Fire rate distribution stats
    active_rates = feature_fire_rate[feature_fire_rate > 0]

    return {
        "mmcs": mmcs,
        "utilization": utilization,
        "dead_features": dead_features,
        "normalized_entropy": normalized_entropy,
        "mean_fire_rate": feature_fire_rate.mean().item(),
        "median_fire_rate": feature_fire_rate.median().item(),
        "max_fire_rate": feature_fire_rate.max().item(),
        "active_features": int((fire_count > 0).sum().item()),
        "total_features": sae.num_latents,
    }


# ── Main Evaluation ──

def run_classification_task(sae, load_fn, model_name, device, max_samples):
    texts, labels, task_name = load_fn(max_samples)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    print("Embedding texts...")
    embeddings = embed_texts(texts, model_name)

    print("SAE encoding...")
    reconstructed, sparse = sae_encode(sae, embeddings, device=device)

    labels = np.array(labels)
    n = len(embeddings)
    split = int(0.8 * n)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, test_idx = idx[:split], idx[split:]

    print("Training probes...")
    raw_result = probe_classification(embeddings[train_idx], labels[train_idx], embeddings[test_idx], labels[test_idx])
    recon_result = probe_classification(reconstructed[train_idx], labels[train_idx], reconstructed[test_idx], labels[test_idx])
    sparse_result = probe_classification(sparse[train_idx], labels[train_idx], sparse[test_idx], labels[test_idx])

    result = {
        "raw_embeddings": raw_result,
        "reconstructed": recon_result,
        "sparse_features": sparse_result,
        "reconstruction_gap": raw_result["accuracy"] - recon_result["accuracy"],
        "feature_quality_gap": raw_result["accuracy"] - sparse_result["accuracy"],
    }

    print(f"  Raw embeddings:  acc={raw_result['accuracy']:.4f}  f1={raw_result['f1_macro']:.4f}")
    print(f"  Reconstructed:   acc={recon_result['accuracy']:.4f}  f1={recon_result['f1_macro']:.4f}  (gap={result['reconstruction_gap']:+.4f})")
    print(f"  Sparse features: acc={sparse_result['accuracy']:.4f}  f1={sparse_result['f1_macro']:.4f}  (gap={result['feature_quality_gap']:+.4f})")

    return task_name, result, embeddings, labels, train_idx, test_idx


def evaluate_sae(sae, model_name, device="cpu", max_samples=5000, suite="standard"):
    results = {}

    # ── Classification probes ──
    tasks = [load_ag_news, load_sst2]
    if suite in ("hard", "full"):
        tasks.extend([load_banking77, load_clinc150])

    last_embeddings = None
    last_labels = None
    last_train_idx = None
    last_test_idx = None

    for load_fn in tasks:
        task_name, result, embeddings, labels, train_idx, test_idx = \
            run_classification_task(sae, load_fn, model_name, device, max_samples)
        results[task_name] = result
        last_embeddings = embeddings
        last_labels = labels
        last_train_idx = train_idx
        last_test_idx = test_idx

    # ── STS-B similarity ──
    texts_pair, labels, task_name = load_stsb(max_samples)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    print("Embedding texts...")
    emb1, emb2 = embed_texts(texts_pair, model_name)

    print("SAE encoding...")
    recon1, sparse1 = sae_encode(sae, emb1, device=device)
    recon2, sparse2 = sae_encode(sae, emb2, device=device)

    labels = np.array(labels)
    n = len(labels)
    idx = np.random.RandomState(42).permutation(n)
    test_idx_sim = idx[int(0.8 * n):]

    raw_sim = probe_similarity(emb1[test_idx_sim], emb2[test_idx_sim], labels[test_idx_sim])
    recon_sim = probe_similarity(recon1[test_idx_sim], recon2[test_idx_sim], labels[test_idx_sim])
    sparse_sim = probe_similarity(sparse1[test_idx_sim], sparse2[test_idx_sim], labels[test_idx_sim])

    results[task_name] = {
        "raw_embeddings": raw_sim, "reconstructed": recon_sim, "sparse_features": sparse_sim,
        "reconstruction_gap": raw_sim["spearman"] - recon_sim["spearman"],
        "feature_quality_gap": raw_sim["spearman"] - sparse_sim["spearman"],
    }
    print(f"  Raw embeddings:  spearman={raw_sim['spearman']:.4f}")
    print(f"  Reconstructed:   spearman={recon_sim['spearman']:.4f}  (gap={results[task_name]['reconstruction_gap']:+.4f})")
    print(f"  Sparse features: spearman={sparse_sim['spearman']:.4f}  (gap={results[task_name]['feature_quality_gap']:+.4f})")

    # ── SciFact Retrieval ──
    if suite in ("hard", "full"):
        print(f"\n{'='*60}")
        print(f"Task: SciFact Retrieval (nDCG@10)")
        print(f"{'='*60}")

        try:
            queries, corpus, qrels = load_scifact()
            query_ids = list(queries.keys())
            corpus_ids = list(corpus.keys())

            print(f"Embedding {len(queries)} queries and {len(corpus)} corpus docs...")
            query_embs = embed_texts(list(queries.values()), model_name)
            corpus_embs = embed_texts(list(corpus.values()), model_name)

            print("SAE encoding...")
            query_recon, query_sparse = sae_encode(sae, query_embs, device=device)
            corpus_recon, corpus_sparse = sae_encode(sae, corpus_embs, device=device)

            raw_ret = eval_retrieval(query_embs, corpus_embs, query_ids, corpus_ids, qrels)
            recon_ret = eval_retrieval(query_recon, corpus_recon, query_ids, corpus_ids, qrels)
            sparse_ret = eval_retrieval(query_sparse, corpus_sparse, query_ids, corpus_ids, qrels)

            results["SciFact Retrieval"] = {
                "raw_embeddings": raw_ret, "reconstructed": recon_ret, "sparse_features": sparse_ret,
                "reconstruction_gap": raw_ret["ndcg@10"] - recon_ret["ndcg@10"],
                "feature_quality_gap": raw_ret["ndcg@10"] - sparse_ret["ndcg@10"],
            }
            print(f"  Raw embeddings:  nDCG@10={raw_ret['ndcg@10']:.4f}  ({raw_ret['n_queries']} queries)")
            print(f"  Reconstructed:   nDCG@10={recon_ret['ndcg@10']:.4f}  (gap={results['SciFact Retrieval']['reconstruction_gap']:+.4f})")
            print(f"  Sparse features: nDCG@10={sparse_ret['ndcg@10']:.4f}  (gap={results['SciFact Retrieval']['feature_quality_gap']:+.4f})")
        except Exception as e:
            print(f"  SciFact failed: {e}")

    # ── k-Sparse Probing ──
    if suite in ("hard", "full") and last_embeddings is not None:
        print(f"\n{'='*60}")
        print(f"Task: k-Sparse Probing (on last classification task)")
        print(f"{'='*60}")

        k_results = {}
        for probe_k in [1, 5, 10, 20, 64]:
            if probe_k > sae.cfg.k:
                continue
            sparse_k = sae_encode_topk_sparse(sae, last_embeddings, probe_k, device=device)
            res = probe_classification(
                sparse_k[last_train_idx], last_labels[last_train_idx],
                sparse_k[last_test_idx], last_labels[last_test_idx]
            )
            k_results[f"k={probe_k}"] = res
            print(f"  k={probe_k:<4}  acc={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}")

        results["k-Sparse Probing"] = k_results

    # ── Feature Quality Metrics ──
    if suite in ("hard", "full"):
        print(f"\n{'='*60}")
        print(f"Feature Quality Metrics")
        print(f"{'='*60}")

        # Use AG News embeddings for feature quality (already loaded)
        ag_texts, _, _ = load_ag_news(max_samples)
        ag_embs = embed_texts(ag_texts, model_name)
        fq = eval_feature_quality(sae, ag_embs, device=device)

        results["Feature Quality"] = fq
        print(f"  MMCS (decoder redundancy): {fq['mmcs']:.4f}  (lower = more distinct features)")
        print(f"  Feature utilization:       {fq['utilization']:.1%}  ({fq['active_features']}/{fq['total_features']} active)")
        print(f"  Dead features:             {fq['dead_features']:.1%}")
        print(f"  Normalized entropy:        {fq['normalized_entropy']:.4f}  (1.0 = uniform, lower = more specific)")
        print(f"  Max fire rate:             {fq['max_fire_rate']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE with downstream probes")
    parser.add_argument("--sae-path", help="Path to local SAE checkpoint directory")
    parser.add_argument("--sae-hub", help="HuggingFace Hub model ID")
    parser.add_argument("--k-expansion", help="k_expansion subfolder for hub models")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--suite", choices=["standard", "hard", "full"], default="standard",
                        help="standard: AG News/SST-2/STS-B. hard: adds BANKING77/CLINC150/SciFact/k-sparse/feature quality")
    parser.add_argument("--output", default="", help="Save results JSON to this path")
    args = parser.parse_args()

    from latentsae import Sae

    if args.sae_path:
        print(f"Loading SAE from disk: {args.sae_path}")
        sae = Sae.load_from_disk(args.sae_path, device=args.device)
    elif args.sae_hub:
        print(f"Loading SAE from hub: {args.sae_hub} / {args.k_expansion}")
        sae = Sae.load_from_hub(args.sae_hub, args.k_expansion, device=args.device)
    else:
        parser.error("Provide --sae-path or --sae-hub")

    print(f"SAE: {sae.num_latents} latents, d_in={sae.d_in}, k={sae.cfg.k}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Suite: {args.suite}")

    results = evaluate_sae(sae, args.embedding_model, device=args.device,
                           max_samples=args.max_samples, suite=args.suite)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for task, r in results.items():
        if task in ("k-Sparse Probing", "Feature Quality"):
            continue
        metric = "accuracy" if "accuracy" in r.get("raw_embeddings", {}) else ("spearman" if "spearman" in r.get("raw_embeddings", {}) else "ndcg@10")
        raw = r["raw_embeddings"][metric]
        recon = r["reconstructed"][metric]
        sparse = r["sparse_features"][metric]
        print(f"  {task:<35} raw={raw:.4f}  recon={recon:.4f} ({r['reconstruction_gap']:+.4f})  sparse={sparse:.4f} ({r['feature_quality_gap']:+.4f})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
