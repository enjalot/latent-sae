"""Build feature_activations.json for a MiniLM SAE.

We don't have the original chunked-120 text on disk, so we re-encode
fresh text from /data/chunks/fineweb-edu-sample-10BT-chunked-500/train/
parquets with all-MiniLM-L6-v2, then forward through the SAE. The
chunk-size mismatch (500 vs 120 at training) is a small distribution
shift but features should still fire on the same concepts; this is
cheaper than re-creating the chunked-120 corpus.

Output: same shape as `extract_pooled_features.py`, ready for
`label_features_ollama` rigor labeling.

Usage:
    python -m experiments.extract_minilm_features \\
        --sae-path /data/hf/.../128_8 \\
        --n-chunks 50000 \\
        --out experiments/results/minilm_external/sae_minilm_128_8/feature_activations.json
"""
import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


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
def encode_texts(texts, device="cuda", batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device=device, cache_folder="/data/hf/sentence-transformers")
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


@torch.no_grad()
def collect_top(sae, embeddings, top_n, batch_size=8192, device="cuda"):
    sae.eval()
    num_latents = sae.num_latents
    feat_heaps = [[] for _ in range(num_latents)]
    n = len(embeddings)
    t0 = time.monotonic()
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = torch.from_numpy(np.ascontiguousarray(embeddings[s:e])).to(device)
        out = sae(batch)
        acts = out.latent_acts.cpu().numpy()
        idxs = out.latent_indices.cpu().numpy()
        for t in range(acts.shape[0]):
            ci = s + t
            for j in range(acts.shape[1]):
                a = float(acts[t, j])
                if a <= 0: continue
                fid = int(idxs[t, j])
                h = feat_heaps[fid]
                if len(h) < top_n:
                    heapq.heappush(h, (a, ci))
                elif a > h[0][0]:
                    heapq.heapreplace(h, (a, ci))
        if s % 16384 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {s+batch_size:,}/{n:,}  ({(s+batch_size)/max(elapsed, 1e-6):.0f} ch/s)", flush=True)
    return feat_heaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-path", required=True)
    ap.add_argument("--n-chunks", type=int, default=50000)
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument("--parquet-dir", default="/data/chunks/fineweb-edu-sample-10BT-chunked-500/train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sae = Sae.load_from_disk(args.sae_path, device="cuda")
    print(f"SAE: {sae.num_latents} latents, d_in={sae.d_in}, k={sae.cfg.k}")

    print(f"loading {args.n_chunks:,} chunks from {args.parquet_dir}")
    texts = load_chunk_texts(args.parquet_dir, args.n_chunks)
    print(f"got {len(texts):,} texts")

    print("encoding with all-MiniLM-L6-v2...")
    embs = encode_texts(texts)
    print(f"embeddings: {embs.shape}")

    print(f"forwarding through SAE, top-{args.top_n} per feature...")
    feat_heaps = collect_top(sae, embs, args.top_n)

    print("serializing...")
    live = []
    per = {}
    for fid, heap in enumerate(feat_heaps):
        if not heap: continue
        srt = sorted(heap, key=lambda x: -x[0])
        per[str(fid)] = [
            {"activation": round(a, 4), "chunk_idx": ci,
             "text": texts[ci], "window": texts[ci]}
            for a, ci in srt
        ]
        live.append(fid)

    payload = {
        "run": Path(args.sae_path).name,
        "checkpoint": Path(args.sae_path).name,
        "sae_type": sae.cfg.sae_type.value if hasattr(sae.cfg.sae_type, "value") else str(sae.cfg.sae_type),
        "num_latents": sae.num_latents,
        "n_chunks_scanned": len(texts),
        "top_n_per_feature": args.top_n,
        "n_live_features": len(live),
        "live_feature_ids": live,
        "embedding_unit": "chunk",
        "features": per,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print(f"live: {len(live)} / {sae.num_latents} ({len(live)/sae.num_latents:.1%})")


if __name__ == "__main__":
    main()
