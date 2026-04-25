"""Embedding-similarity scorer (Neuronpedia-style).

For each labeled feature, compute:
  sim(description, window) over (held-out positive windows) and
                            over (random negative windows).
Aggregate per-feature separation (mean_pos - mean_neg, AUC). No LLM.

Uses mean-pooled mxbai-edge-colbert-v0-32m tokens as a sentence embedder
(64-dim; same ColBERT space the SAE was trained on, which is arguably the
most relevant semantic distance for scoring these features).

Usage:
  python -m experiments.autointerp_embedsim \\
      --run-dir experiments/results/<run> \\
      --dataset fineweb
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLBERT_MODEL_ID = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
TOKEN_MARKER_RE = re.compile(r"<<|>>")

DATASET_PATHS = {
    "fineweb": "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
}


def strip_markers(window: str) -> str:
    return TOKEN_MARKER_RE.sub("", window).strip()


def mean_pool_colbert(texts: list[str], model, batch_size: int = 64,
                      is_query: bool = False) -> np.ndarray:
    """Encode then mean-pool per-token vectors to (n, dim). Vectors are
    already l2-normalized by pylate, so the mean is NOT unit-norm; we
    l2-renormalize after pooling."""
    emb_lists = model.encode(texts, batch_size=batch_size,
                             show_progress_bar=False, is_query=is_query)
    pooled = np.stack([e.mean(axis=0) for e in emb_lists]).astype(np.float32)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8
    return pooled / norms


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann–Whitney U → AUC. Higher = better separation."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_vals = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_vals)) + 1  # dense ranks from 1
    pos_rank_sum = ranks[: len(pos)].sum()
    u = pos_rank_sum - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="fineweb", choices=list(DATASET_PATHS))
    ap.add_argument("--n-features", type=int, default=0,
                    help="0 = score every feature in feature_labels.json; "
                         "otherwise sample this many stratified by label")
    ap.add_argument("--pos-per-feature", type=int, default=10,
                    help="positive windows per feature (from activating top-N, "
                         "skipping the first --desc-windows-seen that labeler saw)")
    ap.add_argument("--desc-windows-seen", type=int, default=10)
    ap.add_argument("--n-negatives", type=int, default=200,
                    help="size of shared negative pool (random chunks, once)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    labels = json.loads((run_dir / "feature_labels.json").read_text())
    acts_data = json.loads((run_dir / "feature_activations.json").read_text())

    # Pick features
    feats_by_label: dict[str, list[str]] = {}
    for fid, rec in labels["labels"].items():
        feats_by_label.setdefault(rec["judgment"], []).append(fid)
    order = ["COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"]

    rng = random.Random(args.seed)
    if args.n_features <= 0:
        sampled = [(fid, rec["judgment"]) for fid, rec in labels["labels"].items()]
    else:
        sampled = []
        per = max(1, args.n_features // max(1, sum(1 for l in order if feats_by_label.get(l))))
        for lbl in order:
            bucket = feats_by_label.get(lbl, []).copy()
            rng.shuffle(bucket)
            sampled.extend([(fid, lbl) for fid in bucket[:per]])
        sampled = sampled[: args.n_features]

    print(f"scoring {len(sampled)} features across {len(set(l for _, l in sampled))} label classes")

    # Sample negatives once (random chunks from the corpus parquets)
    parquets = sorted(Path(DATASET_PATHS[args.dataset]).glob("data-*.parquet"))
    neg_texts: list[str] = []
    for p in parquets:
        df = pd.read_parquet(p, columns=["chunk_text"])
        neg_texts.extend(df["chunk_text"].tolist())
        if len(neg_texts) >= 5 * args.n_negatives:
            break
    rng.shuffle(neg_texts)
    neg_texts = neg_texts[: args.n_negatives]
    print(f"sampled {len(neg_texts)} negative chunks")

    # Load ColBERT
    from pylate import models
    print(f"loading {COLBERT_MODEL_ID} ...")
    model = models.ColBERT(model_name_or_path=COLBERT_MODEL_ID, device=args.device)

    # Encode negatives once
    t0 = time.monotonic()
    neg_vecs = mean_pool_colbert(neg_texts, model, is_query=False)
    print(f"encoded negatives ({neg_vecs.shape}) in {time.monotonic()-t0:.1f}s")

    # Build description list + positive-window list
    desc_texts: list[str] = []
    pos_window_texts: list[list[str]] = []
    for fid, _ in sampled:
        desc_texts.append(labels["labels"][fid]["description"])
        hits = acts_data["features"][fid]
        pool = hits[args.desc_windows_seen:] or hits[-args.pos_per_feature:]
        pos_sel = [strip_markers(h.get("window", "")) for h in pool[: args.pos_per_feature]]
        pos_sel = [w for w in pos_sel if w]
        pos_window_texts.append(pos_sel)

    t0 = time.monotonic()
    desc_vecs = mean_pool_colbert(desc_texts, model, is_query=True)
    print(f"encoded {len(desc_texts)} descriptions in {time.monotonic()-t0:.1f}s")

    # Encode positive windows as ONE big batch for efficiency
    flat_pos: list[str] = []
    pos_slices: list[tuple[int, int]] = []
    cursor = 0
    for p in pos_window_texts:
        pos_slices.append((cursor, cursor + len(p)))
        flat_pos.extend(p)
        cursor += len(p)

    t0 = time.monotonic()
    pos_vecs = mean_pool_colbert(flat_pos, model, is_query=False) if flat_pos else np.zeros((0, 64), dtype=np.float32)
    print(f"encoded {len(flat_pos)} positive windows in {time.monotonic()-t0:.1f}s")

    # Similarities
    # neg_sims: (n_features, n_negatives); pos_sims: list of per-feature arrays
    neg_sims = desc_vecs @ neg_vecs.T  # (F, N)

    per_feature_results = []
    by_label: dict[str, list[float]] = {}
    for i, (fid, lbl) in enumerate(sampled):
        s, e = pos_slices[i]
        if s == e:
            continue
        ps = desc_vecs[i] @ pos_vecs[s:e].T  # (n_pos,)
        ns = neg_sims[i]                     # (N,)
        rec = {
            "feature": fid, "label": lbl,
            "description": desc_texts[i][:200],
            "n_pos": int(len(ps)),
            "mean_pos_sim": float(ps.mean()),
            "mean_neg_sim": float(ns.mean()),
            "std_neg_sim": float(ns.std()),
            "separation": float(ps.mean() - ns.mean()),
            "auc": auc(ps, ns),
        }
        per_feature_results.append(rec)
        by_label.setdefault(lbl, []).append(rec["auc"])

    report = {
        "run": run_dir.name,
        "scorer": "embedding_similarity_colbert_meanpool",
        "model": COLBERT_MODEL_ID,
        "n_features_scored": len(per_feature_results),
        "n_negatives": len(neg_texts),
        "per_label_auc": {k: float(np.mean(v)) for k, v in by_label.items()},
        "per_label_n": {k: len(v) for k, v in by_label.items()},
        "overall_auc": float(np.mean([r["auc"] for r in per_feature_results])) if per_feature_results else 0.0,
        "per_feature": per_feature_results,
    }
    out = run_dir / "feature_embedsim_scores.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\noverall AUC: {report['overall_auc']:.3f}")
    for lbl in order:
        if lbl in by_label:
            print(f"  {lbl:<13} n={len(by_label[lbl]):3d}  AUC={np.mean(by_label[lbl]):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
