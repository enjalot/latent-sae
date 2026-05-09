"""Per-feature top-activating chunks for a pooled-embedding SAE.

For each live feature, gathers the N **chunks** (whole-document
sentence-level units) that activate it most strongly. Output JSON is
shape-compatible with `extract_feature_activations.py` so that the
downstream cheap classifier / topic-cohesion / embedsim scripts work
unchanged — but the unit of analysis is a chunk, not a per-token
window. There is no `token_idx` because the input is one vector per
chunk; we expose the full chunk text under both `text` and `window`.

Source vectors are the disk-backed jina-v5-nano memmaps
(`/data/embeddings/<corpus>-jina-v5-nano/train/data-*.npy`,
fp16, shape (n_chunks, 768)). We do NOT re-embed.

Usage:
    python -m experiments.extract_pooled_features \\
        --sae-dir experiments/results/jina_v5_nano_phase1_24K_oldrecipe_replay4_* \\
        --dataset fineweb \\
        --n-chunks 50000 \\
        --top-n 16
"""
import argparse
import heapq
import json
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


DATASET_PATHS = {
    "fineweb": {
        "vectors": "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train",
        "parquets": "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
    },
    "redpajama": {
        "vectors": "/data/embeddings/RedPajama-Data-V2-sample-10B-chunked-500-jina-v5-nano/train",
        "parquets": "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-500/train",
    },
    "pile": {
        "vectors": "/data/embeddings/pile-uncopyrighted-chunked-500-jina-v5-nano/train",
        "parquets": "/data/chunks/pile-uncopyrighted-chunked-500/train",
    },
}


def list_shards(vec_dir: str) -> list[Path]:
    return sorted(Path(vec_dir).glob("data-*.npy"))


def load_chunk_texts(parquet_dir: str, n_chunks: int) -> list[str]:
    parquets = sorted(Path(parquet_dir).glob("data-*.parquet"))
    frames, got = [], 0
    for p in parquets:
        df = pd.read_parquet(p, columns=["chunk_text"])
        frames.append(df)
        got += len(df)
        if got >= n_chunks:
            break
    df = pd.concat(frames, ignore_index=True).head(n_chunks)
    return df["chunk_text"].tolist()


@torch.no_grad()
def collect_top_chunks(sae: Sae, vec_shards: list[Path], n_chunks: int, top_n: int,
                       batch_size: int = 8192, device: str = "cuda") -> list[list[tuple]]:
    """For each latent, return its top-N (activation, chunk_idx) heap.

    Streams chunk-vectors through the SAE encoder; maintains a per-feature
    min-heap of size top_n.
    """
    sae.eval()
    num_latents = sae.num_latents
    feat_heaps: list[list[tuple]] = [[] for _ in range(num_latents)]

    seen = 0
    t0 = time.monotonic()
    for shard_path in vec_shards:
        if seen >= n_chunks:
            break
        arr = np.load(shard_path, mmap_mode="r")
        n_in_shard = arr.shape[0]
        take = min(n_in_shard, n_chunks - seen)
        for s in range(0, take, batch_size):
            e = min(s + batch_size, take)
            batch = torch.from_numpy(
                np.ascontiguousarray(arr[s:e], dtype=np.float32)
            ).to(device)
            out = sae(batch)
            acts = out.latent_acts.cpu().numpy()
            idxs = out.latent_indices.cpu().numpy()
            chunk_base = seen + s
            n = acts.shape[0]
            for t in range(n):
                ci = chunk_base + t
                for j in range(acts.shape[1]):
                    a = float(acts[t, j])
                    if a <= 0:
                        continue
                    fid = int(idxs[t, j])
                    h = feat_heaps[fid]
                    if len(h) < top_n:
                        heapq.heappush(h, (a, ci))
                    elif a > h[0][0]:
                        heapq.heapreplace(h, (a, ci))
        seen += take
        elapsed = time.monotonic() - t0
        rate = seen / max(elapsed, 1e-6)
        print(f"  {seen:,}/{n_chunks:,} chunks  ({rate:.0f} ch/s)", flush=True)
    return feat_heaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--dataset", default="fineweb", choices=list(DATASET_PATHS))
    ap.add_argument("--n-chunks", type=int, default=50000)
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    candidates = sorted(Path(p) for p in glob(args.sae_dir))
    if not candidates:
        raise FileNotFoundError(f"no matches for {args.sae_dir}")
    run_dir = candidates[0]
    ckpt = next((p for p in (run_dir / "checkpoints").glob("*")
                 if p.is_dir() and (p / "cfg.json").exists()), None)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    print(f"run: {run_dir.name}")
    print(f"checkpoint: {ckpt.name}")

    sae = Sae.load_from_disk(ckpt, device=args.device)
    print(f"SAE: {sae.cfg.sae_type.value}, num_latents={sae.num_latents}, d_in={sae.W_dec.shape[-1]}")

    paths = DATASET_PATHS[args.dataset]
    vec_shards = list_shards(paths["vectors"])
    if not vec_shards:
        raise FileNotFoundError(f"no .npy shards in {paths['vectors']}")
    total_avail = sum(np.load(p, mmap_mode="r").shape[0] for p in vec_shards)
    n_chunks = min(args.n_chunks, total_avail)
    print(f"vectors: {len(vec_shards)} shards, {total_avail:,} chunks total, scanning {n_chunks:,}")

    feat_heaps = collect_top_chunks(sae, vec_shards, n_chunks, args.top_n,
                                    args.batch_size, args.device)

    print(f"loading {n_chunks:,} chunk texts...")
    chunk_texts = load_chunk_texts(paths["parquets"], n_chunks)

    print("serializing feature activations...")
    live_features = []
    per_feature = {}
    for fid, heap in enumerate(feat_heaps):
        if not heap:
            continue
        sorted_hits = sorted(heap, key=lambda x: -x[0])
        entries = []
        for act, ci in sorted_hits:
            text = chunk_texts[ci] if ci < len(chunk_texts) else ""
            entries.append({
                "activation": round(act, 4),
                "chunk_idx": ci,
                "text": text,
                # `window` for downstream-script compatibility (cheap classifier
                # + topic_cohesion read this key). For pooled SAE, the "window"
                # is just the full chunk text.
                "window": text,
            })
        per_feature[str(fid)] = entries
        live_features.append(fid)

    out_path = Path(args.out) if args.out else run_dir / "feature_activations.json"
    payload = {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "sae_type": sae.cfg.sae_type.value,
        "num_latents": sae.num_latents,
        "n_chunks_scanned": n_chunks,
        "top_n_per_feature": args.top_n,
        "n_live_features": len(live_features),
        "live_feature_ids": live_features,
        "embedding_unit": "chunk",
        "features": per_feature,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    print(f"live features: {len(live_features)} / {sae.num_latents} "
          f"({len(live_features)/sae.num_latents:.1%})")


if __name__ == "__main__":
    main()
