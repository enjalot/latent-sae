"""Build three UMAPs per SAE for feature visualization.

For each live feature in an SAE, compute three 2D coords:
  1. activation_centroid (act): mean of ColBERT input vectors at the
     top-N activating positions for that feature (so features that
     fire on similar inputs cluster together).
  2. decoder (dec): UMAP of the W_dec rows (per-feature output direction).
  3. encoder (enc): UMAP of the encoder.weight rows (per-feature
     input-projection direction).

Saves a single parquet per SAE with columns:
  feature, judgment, description, max_activation, n_hits,
  cohesion_gap, embedsim_auc,
  act_x, act_y, dec_x, dec_y, enc_x, enc_y

Usage:
    python -m experiments.build_feature_umaps \\
        --sae-dir <run dir> \\
        --activation-source /data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train \\
        --out experiments/results/<run>/feature_umaps.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


def find_sae_ckpt(run_dir: Path) -> Path | None:
    for p in sorted((run_dir / "checkpoints").glob("*"), key=lambda x: x.name):
        if p.is_dir() and (p / "cfg.json").exists():
            return p
    return None


def per_feature_activation_centroid(activations: dict, vec_path: Path,
                                     d_in: int, top_n: int = 20) -> dict[int, np.ndarray]:
    """For each feature, compute the mean of ColBERT vectors at its top-N
    activating positions.

    activations["features"][fid] is a list of dicts with chunk_idx, token_idx,
    activation, window. We use the token's GLOBAL index (chunk_offsets[chunk_idx]
    + token_idx) into the cached vector memmap.
    """
    print(f"loading vector memmap from {vec_path}")
    # Vectors are stored as a single npy or a sequence of data-*.npy
    vec_files = sorted(vec_path.glob("data-*.npy"))
    if not vec_files:
        # try monolithic
        vec_files = [vec_path / "data.npy"]
    vecs = np.load(vec_files[0], mmap_mode="r")
    offsets = np.load(vec_path / "chunk_offsets.npy")
    print(f"  vectors: {vecs.shape}  offsets: {offsets.shape}")

    centroids: dict[int, np.ndarray] = {}
    feats = activations["features"]
    print(f"computing centroids for {len(feats)} features...")
    t0 = time.monotonic()
    for i, (fid, hits) in enumerate(feats.items()):
        sel = hits[:top_n]
        if not sel:
            continue
        vecs_sel = []
        for h in sel:
            chunk = h["chunk_idx"]; tok = h["token_idx"]
            global_idx = int(offsets[chunk]) + int(tok)
            if global_idx < 0 or global_idx >= vecs.shape[0]:
                continue
            vecs_sel.append(np.asarray(vecs[global_idx], dtype=np.float32))
        if vecs_sel:
            centroids[int(fid)] = np.mean(vecs_sel, axis=0)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(feats)}  ({time.monotonic()-t0:.1f}s elapsed)")
    print(f"  done in {time.monotonic()-t0:.1f}s, {len(centroids)} centroids")
    return centroids


def umap_2d(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
            random_state: int = 42, label: str = "") -> np.ndarray:
    import umap
    print(f"UMAP {label}: input {X.shape} → 2D ...")
    t0 = time.monotonic()
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state,
                        n_jobs=1)
    Y = reducer.fit_transform(X)
    print(f"  done in {time.monotonic()-t0:.1f}s")
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--activation-source",
                    default="/data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train",
                    help="path to the dataset's vector cache (chunk_offsets.npy + data-*.npy)")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rd = Path(args.sae_dir)
    ckpt = find_sae_ckpt(rd)
    print(f"loading SAE from {ckpt}")
    sae = Sae.load_from_disk(ckpt, device="cpu")  # we only need W_dec / encoder.weight on CPU
    num_latents = sae.num_latents
    d_in = sae.d_in
    print(f"  num_latents={num_latents}, d_in={d_in}")

    # Live features list (whose centroids we can compute)
    activations = json.loads((rd / "feature_activations.json").read_text())
    live_ids = activations["live_feature_ids"]
    print(f"live feature count: {len(live_ids)}")

    centroids = per_feature_activation_centroid(
        activations, Path(args.activation_source), d_in, top_n=args.top_n)

    # Build per-feature matrices in CONSISTENT row order
    common_ids = [fid for fid in live_ids if fid in centroids]
    common_ids.sort()
    print(f"features with centroids: {len(common_ids)}")

    # Activation centroid matrix
    F_act = np.stack([centroids[fid] for fid in common_ids], axis=0)
    # Decoder rows (W_dec[fid])
    W_dec = sae.W_dec.detach().cpu().numpy()
    F_dec = W_dec[common_ids]
    # Encoder rows (encoder.weight[fid])
    W_enc = sae.encoder.weight.detach().cpu().numpy()
    F_enc = W_enc[common_ids]

    # Three UMAPs
    Y_act = umap_2d(F_act, n_neighbors=args.n_neighbors, min_dist=args.min_dist, label="activation centroid")
    Y_dec = umap_2d(F_dec, n_neighbors=args.n_neighbors, min_dist=args.min_dist, label="decoder")
    Y_enc = umap_2d(F_enc, n_neighbors=args.n_neighbors, min_dist=args.min_dist, label="encoder")

    # Pull labels + scorer outputs to enrich
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
    for i, fid in enumerate(common_ids):
        rec = {
            "feature": fid,
            "judgment": labels.get(str(fid), {}).get("judgment", ""),
            "description": labels.get(str(fid), {}).get("description", ""),
            "max_activation": float(activations["features"][str(fid)][0]["activation"])
                if activations["features"].get(str(fid)) else 0.0,
            "n_hits": len(activations["features"].get(str(fid), [])),
            "cohesion_gap": float(coh_per.get(str(fid), {}).get("gap", 0.0)),
            "embedsim_auc": float(es_per.get(str(fid), {}).get("auc", 0.5)),
            "act_x": float(Y_act[i, 0]), "act_y": float(Y_act[i, 1]),
            "dec_x": float(Y_dec[i, 0]), "dec_y": float(Y_dec[i, 1]),
            "enc_x": float(Y_enc[i, 0]), "enc_y": float(Y_enc[i, 1]),
        }
        rows.append(rec)

    out = Path(args.out) if args.out else rd / "feature_umaps.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({df.shape}, {out.stat().st_size // 1024} KB)")
    print(f"\nfirst rows:")
    print(df.head(3))


if __name__ == "__main__":
    main()
