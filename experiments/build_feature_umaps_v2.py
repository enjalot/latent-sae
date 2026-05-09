"""Feature UMAPs v2 — addresses the ColBERT 64-dim hypersphere concentration.

Adds two new representations alongside the original three:
  - act_centered: ColBERT centroid minus the dataset-wide mean (whitened)
  - act_minilm:   per-feature centroid in MiniLM-L6-v2 sentence-embedding
                  space (384-dim) — embed each feature's top-N activating
                  WINDOWS (the rendered text) and average

Also switches to cosine-metric UMAP for all variants. Cosine respects the
unit-sphere geometry better than Euclidean for ColBERT vectors.

Usage:
    python -m experiments.build_feature_umaps_v2 \\
        --sae-dir <run dir> \\
        --activation-source /data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402

MARKER_RE = re.compile(r"<<|>>")


def find_sae_ckpt(run_dir: Path) -> Path | None:
    for p in sorted((run_dir / "checkpoints").glob("*"), key=lambda x: x.name):
        if p.is_dir() and (p / "cfg.json").exists():
            return p
    return None


def per_feature_colbert_centroid(activations: dict, vec_path: Path,
                                   top_n: int = 20) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Returns (per-feature centroid dict, dataset mean of all sampled vectors)."""
    vec_files = sorted(vec_path.glob("data-*.npy"))
    vecs = np.load(vec_files[0], mmap_mode="r")
    offsets = np.load(vec_path / "chunk_offsets.npy")

    centroids: dict[int, np.ndarray] = {}
    feats = activations["features"]
    print(f"computing ColBERT centroids for {len(feats)} features...")
    t0 = time.monotonic()
    all_used = []
    for fid, hits in feats.items():
        sel = hits[:top_n]
        if not sel:
            continue
        vs = []
        for h in sel:
            gi = int(offsets[h["chunk_idx"]]) + int(h["token_idx"])
            if 0 <= gi < vecs.shape[0]:
                vs.append(np.asarray(vecs[gi], dtype=np.float32))
        if vs:
            arr = np.stack(vs)
            centroids[int(fid)] = arr.mean(axis=0)
            all_used.append(arr)
    used = np.concatenate(all_used, axis=0)
    dataset_mean = used.mean(axis=0)
    print(f"  done in {time.monotonic() - t0:.1f}s, {len(centroids)} centroids; "
          f"dataset mean norm = {np.linalg.norm(dataset_mean):.3f}")
    return centroids, dataset_mean


def per_feature_minilm_centroid(activations: dict, top_n: int = 10) -> dict[int, np.ndarray]:
    """For each feature, encode the rendered windows of top-N activating
    positions with MiniLM-L6-v2 and average."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2",
                                 cache_folder="/data/hf/sentence-transformers",
                                 device="cuda")
    feats = activations["features"]
    fid_order = []
    flat_texts = []
    spans = []
    for fid, hits in feats.items():
        windows = [MARKER_RE.sub("", h.get("window", "")).strip() for h in hits[:top_n]]
        windows = [w for w in windows if w]
        if not windows:
            continue
        s = len(flat_texts)
        flat_texts.extend(windows)
        e = len(flat_texts)
        spans.append((int(fid), s, e))
        fid_order.append(int(fid))
    print(f"encoding {len(flat_texts)} windows with MiniLM...")
    t0 = time.monotonic()
    vecs = model.encode(flat_texts, batch_size=256, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True)
    print(f"  done in {time.monotonic() - t0:.1f}s, vec shape={vecs.shape}")

    centroids = {}
    for fid, s, e in spans:
        c = vecs[s:e].mean(axis=0)
        n = np.linalg.norm(c)
        if n > 1e-8:
            c = c / n
        centroids[fid] = c
    return centroids


def umap_2d(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1,
            metric: str = "cosine", random_state: int = 42, label: str = "") -> np.ndarray:
    import umap
    print(f"UMAP {label}: input {X.shape} (metric={metric}) → 2D ...")
    t0 = time.monotonic()
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric=metric,
                        random_state=random_state, n_jobs=1)
    Y = reducer.fit_transform(X)
    print(f"  done in {time.monotonic() - t0:.1f}s")
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--activation-source",
                    default="/data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--minilm-top-n", type=int, default=10)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.sae_dir)
    ckpt = find_sae_ckpt(rd)
    print(f"loading SAE from {ckpt}")
    sae = Sae.load_from_disk(ckpt, device="cpu")

    activations = json.loads((rd / "feature_activations.json").read_text())
    live_ids = activations["live_feature_ids"]

    # ColBERT centroids (raw + centered)
    cb_cent, dataset_mean = per_feature_colbert_centroid(
        activations, Path(args.activation_source), top_n=args.top_n)

    # MiniLM centroids
    ml_cent = per_feature_minilm_centroid(activations, top_n=args.minilm_top_n)

    common = sorted([fid for fid in live_ids if fid in cb_cent and fid in ml_cent])
    print(f"\ncommon features (both representations): {len(common)}")

    F_act_raw = np.stack([cb_cent[fid] for fid in common], axis=0)
    F_act_cent = F_act_raw - dataset_mean
    F_act_ml = np.stack([ml_cent[fid] for fid in common], axis=0)

    W_dec = sae.W_dec.detach().cpu().numpy()
    F_dec = W_dec[common]
    W_enc = sae.encoder.weight.detach().cpu().numpy()
    F_enc = W_enc[common]

    # All five UMAPs, cosine metric
    Y_act_raw = umap_2d(F_act_raw, n_neighbors=args.n_neighbors,
                         min_dist=args.min_dist, metric="cosine",
                         label="act_raw (ColBERT)")
    Y_act_cent = umap_2d(F_act_cent, n_neighbors=args.n_neighbors,
                          min_dist=args.min_dist, metric="cosine",
                          label="act_cent (whitened ColBERT)")
    Y_act_ml = umap_2d(F_act_ml, n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist, metric="cosine",
                        label="act_ml (MiniLM)")
    Y_dec = umap_2d(F_dec, n_neighbors=args.n_neighbors,
                     min_dist=args.min_dist, metric="cosine",
                     label="decoder W_dec")
    Y_enc = umap_2d(F_enc, n_neighbors=args.n_neighbors,
                     min_dist=args.min_dist, metric="cosine",
                     label="encoder W_enc")

    labels_path = rd / "feature_labels.json"
    if labels_path.is_symlink():
        labels_path = labels_path.resolve()
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())["labels"]
    coh_per = {}
    coh_path = rd / "feature_topic_cohesion_scores.json"
    if coh_path.exists():
        for r in json.loads(coh_path.read_text()).get("per_feature", []):
            coh_per[r["feature"]] = r
    es_per = {}
    es_path = rd / "feature_embedsim_scores.json"
    if es_path.exists():
        for r in json.loads(es_path.read_text()).get("per_feature", []):
            es_per[r["feature"]] = r

    rows = []
    for i, fid in enumerate(common):
        rec = {
            "feature": fid,
            "judgment": labels.get(str(fid), {}).get("judgment", ""),
            "description": labels.get(str(fid), {}).get("description", ""),
            "max_activation": float(activations["features"][str(fid)][0]["activation"])
                if activations["features"].get(str(fid)) else 0.0,
            "n_hits": len(activations["features"].get(str(fid), [])),
            "cohesion_gap": float(coh_per.get(str(fid), {}).get("gap", 0.0)),
            "embedsim_auc": float(es_per.get(str(fid), {}).get("auc", 0.5)),
            # Five UMAPs
            "act_x": float(Y_act_raw[i, 0]),  "act_y": float(Y_act_raw[i, 1]),
            "actc_x": float(Y_act_cent[i, 0]), "actc_y": float(Y_act_cent[i, 1]),
            "actm_x": float(Y_act_ml[i, 0]),   "actm_y": float(Y_act_ml[i, 1]),
            "dec_x": float(Y_dec[i, 0]),       "dec_y": float(Y_dec[i, 1]),
            "enc_x": float(Y_enc[i, 0]),       "enc_y": float(Y_enc[i, 1]),
        }
        rows.append(rec)

    out = Path(args.out) if args.out else rd / "feature_umaps_v2.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({df.shape}, {out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
