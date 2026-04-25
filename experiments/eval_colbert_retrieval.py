"""ColBERT MaxSim retrieval eval for SAE checkpoints.

Compares raw mxbai-colbert MaxSim nDCG@10 / MRR@10 to SAE-reconstructed
MaxSim on a BEIR-style retrieval dataset. Embeddings for the dataset are
computed once and cached under /data/embeddings/beir/<dataset>-<model_slug>/.

Scoring semantics (ColBERT late interaction):
    score(Q, D) = sum_{q in Q} max_{d in D} <q, d>
where Q is a (nq_tokens, dim) query-encoded matrix and D is a
(nd_tokens, dim) document-encoded matrix. Both sides l2-normalized by pylate.

Usage:
    python -m experiments.eval_colbert_retrieval \\
        --dataset scifact \\
        --sae-dir experiments/results/colbert_mxbai_phase5__k4_*  \\
        --sae-dir experiments/results/colbert_mxbai_phase4__k8_expansion_factor2_* \\
        ...

The script treats multiple --sae-dir args as separate runs; results are
written per-run plus an aggregate comparison table.
"""
import argparse
import json
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402

MODEL_ID = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
MODEL_SLUG = "mxbai-edge-32m"
DATASETS = {
    "scifact": {"path": "mteb/scifact"},
    "trec-covid": {"path": "mteb/trec-covid"},
}


# ---------- Data loading ----------

def load_beir(dataset_key: str):
    from datasets import load_dataset
    cfg = DATASETS[dataset_key]
    queries_ds = load_dataset(cfg["path"], "queries", split="queries")
    corpus_ds = load_dataset(cfg["path"], "corpus", split="corpus")
    qrels_ds = load_dataset(cfg["path"], "default", split="test")

    queries = {row["_id"]: row["text"] for row in queries_ds}
    corpus = {row["_id"]: (row.get("title", "") + " " + row["text"]).strip()
              for row in corpus_ds}
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    # Restrict queries to those with judgments
    queries = {qid: txt for qid, txt in queries.items() if qid in qrels}
    return queries, corpus, qrels


# ---------- ColBERT encoding + cache ----------

def _pack(emb_list) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a list of (n_tokens, dim) arrays to flat matrix + offsets."""
    lens = np.array([e.shape[0] for e in emb_list], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lens)))
    vecs = np.concatenate(emb_list, axis=0).astype(np.float16, copy=False)
    return vecs, offsets


def _unpack(vecs: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [vecs[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def embed_or_load(name: str, ids: list[str], texts: list[str], out_dir: Path,
                  is_query: bool, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode via mxbai-colbert and cache to out_dir/<name>_{vectors,offsets,ids}.npy."""
    vec_path = out_dir / f"{name}_vectors.npy"
    off_path = out_dir / f"{name}_offsets.npy"
    id_path = out_dir / f"{name}_ids.json"
    if vec_path.exists() and off_path.exists() and id_path.exists():
        print(f"  cache hit: {name}")
        vecs = np.load(vec_path, mmap_mode="r")
        offsets = np.load(off_path)
        cached_ids = json.loads(id_path.read_text())
        return vecs, offsets, cached_ids

    from pylate import models
    print(f"  encoding {len(ids)} {name} (is_query={is_query}) ...")
    t = time.monotonic()
    model = models.ColBERT(model_name_or_path=MODEL_ID, device="cuda")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       is_query=is_query)
    del model
    torch.cuda.empty_cache()
    vecs, offsets = _pack(emb)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vecs)
    np.save(off_path, offsets)
    id_path.write_text(json.dumps(ids))
    print(f"  {name}: {vecs.shape[0]:,} total tokens across {len(ids)} items "
          f"({time.monotonic() - t:.1f}s)")
    return vecs, offsets, ids


# ---------- SAE reconstruction over cached token matrices ----------

@torch.no_grad()
def sae_reconstruct(sae: Sae, vecs: np.ndarray, batch_size: int = 8192,
                    device: str = "cuda") -> np.ndarray:
    """Feed (N, dim) matrix through SAE; return reconstructed (N, dim) fp16."""
    sae.eval()
    n = vecs.shape[0]
    out = np.empty_like(vecs)
    for s in range(0, n, batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(vecs[s:s + batch_size], dtype=np.float32)).to(device)
        recon = sae(batch).sae_out
        out[s:s + batch_size] = recon.cpu().numpy().astype(np.float16)
    return out


# ---------- MaxSim + nDCG ----------

@torch.no_grad()
def maxsim_scores(q_vecs: np.ndarray, q_off: np.ndarray,
                  d_vecs: np.ndarray, d_off: np.ndarray,
                  device: str = "cuda") -> np.ndarray:
    """Dense MaxSim: returns (n_queries, n_docs) score matrix.

    For each query q_i with tokens Q_i (nq_i, dim) and each doc d_j with
    tokens D_j (nd_j, dim): score = sum_{t in Q_i} max_{s in D_j} <Q_i[t], D_j[s]>.
    """
    D = torch.from_numpy(np.ascontiguousarray(d_vecs, dtype=np.float32)).to(device)
    n_q = len(q_off) - 1
    n_d = len(d_off) - 1
    d_off_t = torch.tensor(d_off, device=device)
    # Per-doc token count used for scatter-max
    scores = torch.zeros((n_q, n_d), device=device)
    # Segment IDs for each doc token
    doc_ids = torch.zeros(D.shape[0], dtype=torch.long, device=device)
    for j in range(n_d):
        doc_ids[d_off[j]:d_off[j + 1]] = j

    for i in range(n_q):
        q_tokens = torch.from_numpy(
            np.ascontiguousarray(q_vecs[q_off[i]:q_off[i + 1]], dtype=np.float32)
        ).to(device)  # (nq_i, dim)
        # (nq_i, total_doc_tokens)
        sims = q_tokens @ D.T
        # Per-doc max of each query token → (nq_i, n_d)
        # We bucket by doc id using scatter_reduce amax
        qmax = torch.full((q_tokens.shape[0], n_d), float("-inf"),
                          device=device)
        qmax.scatter_reduce_(1, doc_ids.expand(q_tokens.shape[0], -1),
                             sims, reduce="amax", include_self=True)
        # Sum across query tokens
        scores[i] = qmax.sum(dim=0)
    return scores.cpu().numpy()


def spearman_vs_raw(raw_scores: np.ndarray, recon_scores: np.ndarray) -> float:
    """Mean Spearman correlation between per-query raw and reconstructed score
    vectors — high value means ranking of documents is preserved even if
    absolute MaxSim magnitudes differ."""
    from scipy.stats import spearmanr
    rhos = []
    for i in range(raw_scores.shape[0]):
        rho, _ = spearmanr(raw_scores[i], recon_scores[i])
        if rho == rho:  # skip NaN
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0


def ndcg_at_k(scores: np.ndarray, q_ids: list[str], d_ids: list[str],
              qrels: dict, k: int = 10) -> dict:
    """Compute nDCG@k and MRR@k averaged over queries."""
    doc_id_idx = {cid: j for j, cid in enumerate(d_ids)}
    ndcg_vals, rr_vals = [], []
    for i, qid in enumerate(q_ids):
        row = scores[i]
        order = np.argsort(-row)  # descending
        relevant = qrels.get(qid, {})
        if not relevant:
            continue
        dcg = 0.0
        rr = 0.0
        for rank, j in enumerate(order[:k]):
            cid = d_ids[j]
            g = relevant.get(cid, 0)
            if g > 0:
                dcg += (2 ** g - 1) / np.log2(rank + 2)
                if rr == 0.0:
                    rr = 1.0 / (rank + 1)
        # Ideal DCG: sort relevance descending
        ideal_grades = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / np.log2(r + 2)
                   for r, g in enumerate(ideal_grades))
        ndcg_vals.append(dcg / idcg if idcg > 0 else 0.0)
        rr_vals.append(rr)
    return {
        f"ndcg@{k}": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        f"mrr@{k}":  float(np.mean(rr_vals))   if rr_vals   else 0.0,
        "n_queries": len(ndcg_vals),
    }


# ---------- Driver ----------

def find_sae_ckpt(run_dir: Path) -> Path | None:
    for p in (run_dir / "checkpoints").glob("*"):
        if p.is_dir() and (p / "cfg.json").exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=list(DATASETS))
    ap.add_argument("--sae-dir", action="append", default=[],
                    help="One or more run directories (globs OK)")
    ap.add_argument("--cache-root", default="/data/embeddings/beir")
    ap.add_argument("--results-out", default=None,
                    help="write aggregate JSON here (default: <cache_dir>/retrieval_results.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # Expand globs
    run_dirs: list[Path] = []
    for pattern in args.sae_dir:
        expanded = sorted(Path(p) for p in glob(pattern))
        if expanded:
            run_dirs.extend(expanded)
        else:
            p = Path(pattern)
            if p.exists():
                run_dirs.append(p)

    # 1. Load dataset
    print(f"\n=== {args.dataset} ===")
    queries, corpus, qrels = load_beir(args.dataset)
    q_ids = sorted(queries.keys())
    d_ids = sorted(corpus.keys())
    print(f"  {len(q_ids)} queries, {len(d_ids)} docs, {sum(len(v) for v in qrels.values())} qrels")

    # 2. Embed (with cache)
    cache_dir = Path(args.cache_root) / f"{args.dataset}-{MODEL_SLUG}"
    q_vecs, q_off, _ = embed_or_load("queries", q_ids, [queries[qid] for qid in q_ids],
                                     cache_dir, is_query=True)
    d_vecs, d_off, _ = embed_or_load("corpus", d_ids, [corpus[cid] for cid in d_ids],
                                     cache_dir, is_query=False)

    # 3. Baseline MaxSim on raw
    print("\n-- raw ColBERT baseline --")
    t = time.monotonic()
    raw_scores = maxsim_scores(q_vecs, q_off, d_vecs, d_off, device=args.device)
    metrics = ndcg_at_k(raw_scores, q_ids, d_ids, qrels, k=args.k)
    print(f"  {metrics}  ({time.monotonic() - t:.1f}s)")
    results = [{"run": "raw_colbert", "run_dir": None, "spearman_vs_raw": 1.0, **metrics}]

    # 4. SAE-reconstructed scoring
    for rd in run_dirs:
        ckpt = find_sae_ckpt(rd)
        if ckpt is None:
            print(f"SKIP {rd.name}: no checkpoint")
            continue
        print(f"\n-- {rd.name} / {ckpt.name} --")
        sae = Sae.load_from_disk(ckpt, device=args.device)
        t = time.monotonic()
        q_recon = sae_reconstruct(sae, q_vecs, device=args.device)
        d_recon = sae_reconstruct(sae, d_vecs, device=args.device)
        sc = maxsim_scores(q_recon, q_off, d_recon, d_off, device=args.device)
        m = ndcg_at_k(sc, q_ids, d_ids, qrels, k=args.k)
        rho = spearman_vs_raw(raw_scores, sc)
        m["spearman_vs_raw"] = rho
        wall = time.monotonic() - t
        print(f"  {m}  ({wall:.1f}s)")
        results.append({"run": rd.name, "run_dir": str(rd),
                        "checkpoint": ckpt.name, **m})
        del sae
        torch.cuda.empty_cache()

    # 5. Aggregate report
    out_path = Path(args.results_out) if args.results_out else cache_dir / "retrieval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"dataset": args.dataset, "k": args.k,
                                     "results": results}, indent=2))
    print(f"\nwrote {out_path}")
    print(f"\n=== SUMMARY ({args.dataset}, nDCG@{args.k}) ===")
    print(f"{'run':70}  {'nDCG':>6}  {'MRR':>6}  {'ρ_rank':>7}  {'Δ vs raw':>9}")
    raw_ndcg = results[0][f"ndcg@{args.k}"]
    for r in results:
        nd = r[f"ndcg@{args.k}"]
        mrr = r[f"mrr@{args.k}"]
        rho = r.get("spearman_vs_raw", float("nan"))
        delta = f"{nd - raw_ndcg:+.4f}"
        rho_s = f"{rho:.3f}" if rho == rho else "  -  "
        print(f"{r['run']:70}  {nd:>6.4f}  {mrr:>6.4f}  {rho_s:>7}  {delta:>9}")


if __name__ == "__main__":
    main()
