"""Embed the first N chunks of a chunked-parquet dir with a pylate ColBERT
model and write the token-embedding memmap layout consumed by
extract_feature_activations.py:

    <out>/data-00000.npy        (total_tokens, d) fp16
    <out>/chunk_offsets.npy     (n_chunks + 1,) int64 cumulative token counts

Usage:
    python -m experiments.embed_chunks_pylate \\
        --parquet-dir /data/chunks/fineweb-edu-sample-10BT-chunked-500/train \\
        --model-id answerdotai/answerai-colbert-small-v1 \\
        --out /data/embeddings/fineweb-edu-sample-10BT-chunked-500-answerai-small/train \\
        --n-chunks 10000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-chunks", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    from pylate import models
    model = models.ColBERT(model_name_or_path=args.model_id, device=args.device,
                           trust_remote_code=args.trust_remote_code)

    texts = load_chunk_texts(args.parquet_dir, args.n_chunks)
    texts = [t if t and t.strip() else " " for t in texts]
    print(f"embedding {len(texts):,} chunks with {args.model_id}")

    arrs, offsets = [], [0]
    for s in range(0, len(texts), args.batch_size):
        embs = model.encode(
            texts[s:s + args.batch_size],
            batch_size=args.batch_size,
            is_query=False,
            show_progress_bar=False,
            convert_to_numpy=False,
        )
        for e in embs:
            arrs.append(e.to(dtype=torch.float16).cpu().numpy())
            offsets.append(offsets[-1] + e.shape[0])
        if (s // args.batch_size) % 20 == 0:
            print(f"  {s + len(embs):,}/{len(texts):,} chunks, "
                  f"{offsets[-1]:,} tokens", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate(arrs, axis=0)
    np.save(out / "data-00000.npy", flat)
    np.save(out / "chunk_offsets.npy", np.asarray(offsets, dtype=np.int64))
    print(f"wrote {flat.shape} fp16 -> {out}/data-00000.npy")
    print(f"wrote offsets ({len(offsets)}) -> {out}/chunk_offsets.npy")


if __name__ == "__main__":
    main()
