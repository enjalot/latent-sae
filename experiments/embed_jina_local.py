"""Embed a chunked-text corpus with jina-v5-nano (or small) and save
fp16 vectors to disk in a layout compatible with downstream SAE
training.

Output layout (one .npy per source parquet shard, preserves shard
alignment so a downstream tool can map a vector index back to its
source row):
  /data/embeddings/<corpus>-jina-v5-nano-500/train/
      data-00000-of-NNNNN.npy        # (n_chunks_in_shard, 768) fp16
      ...
      manifest.json                   # per-shard counts + total
      embedding_meta.json             # model, max_seq_length, batch, etc

Usage:
  python -m experiments.embed_jina_local \\
      --source /data/chunks/fineweb-edu-sample-10BT-chunked-500/train \\
      --output /data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano \\
      --model jinaai/jina-embeddings-v5-text-nano-retrieval \\
      --batch-size 32 --max-seq-length 512
"""
import argparse
import json
import os
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def embed_shard(parquet_path: Path, out_path: Path, model, batch_size: int,
                 max_seq_length: int, dtype: str = "float16") -> dict:
    df = pd.read_parquet(parquet_path, columns=["chunk_text"])
    texts = df["chunk_text"].tolist()
    n = len(texts)
    if out_path.exists():
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape[0] == n:
            return {"shard": parquet_path.name, "n": n, "skipped": True}
    t0 = time.monotonic()
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True)
    elapsed = time.monotonic() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vecs.astype(dtype))
    return {"shard": parquet_path.name, "n": n, "elapsed_s": elapsed,
             "throughput_chunks_per_s": n / elapsed if elapsed > 0 else 0,
             "out_path": str(out_path), "out_size_mib": out_path.stat().st_size // (1024 * 1024)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="parquet train dir, e.g. /data/chunks/<corpus>/train")
    ap.add_argument("--output", required=True,
                    help="output dir for /train/data-NNNNN.npy")
    ap.add_argument("--model", default="jinaai/jina-embeddings-v5-text-nano-retrieval")
    ap.add_argument("--slug", default=None,
                    help="model slug for metadata (e.g. jina-v5-nano)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="(testing) limit to first N shards")
    ap.add_argument("--glob", default=None,
                    help="parquet filename glob (default tries 'data-*.parquet' "
                         "then '*.parquet' if none match — covers both English "
                         "data-NNNNN-of-MMMMM.parquet and multilingual "
                         "000_NNNNN.parquet conventions)")
    args = ap.parse_args()

    src = Path(args.source); out = Path(args.output) / "train"
    out.mkdir(parents=True, exist_ok=True)
    if args.glob:
        parquets = sorted(src.glob(args.glob))
    else:
        parquets = sorted(src.glob("data-*.parquet"))
        if not parquets:
            parquets = sorted(src.glob("*.parquet"))
    if args.limit_shards:
        parquets = parquets[: args.limit_shards]
    print(f"source: {src}")
    print(f"output: {out}")
    print(f"shards: {len(parquets)}")
    print(f"model:  {args.model}")
    print(f"batch:  {args.batch_size}  max_seq: {args.max_seq_length}  dtype: {args.dtype}")

    # Set up model
    os.environ["HF_HOME"] = "/data/hf"
    from sentence_transformers import SentenceTransformer
    print(f"loading model...")
    model = SentenceTransformer(args.model,
                                 cache_folder="/data/hf/sentence-transformers",
                                 device=args.device, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    shard_log = []
    total_start = time.monotonic()
    total_n = 0
    for i, p in enumerate(parquets):
        # Match output filename to source numbering
        out_path = out / (p.stem + ".npy")
        result = embed_shard(p, out_path, model,
                              batch_size=args.batch_size,
                              max_seq_length=args.max_seq_length,
                              dtype=args.dtype)
        shard_log.append(result)
        total_n += result["n"]
        elapsed = time.monotonic() - total_start
        rate = total_n / elapsed if elapsed > 0 else 0
        eta_total = sum(r["n"] for r in shard_log) / rate if rate > 0 else 0
        # Save manifest after each shard (resilient to crashes)
        manifest = {
            "model": args.model,
            "slug": args.slug or args.model.split("/")[-1],
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "source": str(src),
            "n_shards": len(parquets),
            "shards_completed": len(shard_log),
            "total_chunks_so_far": total_n,
            "shards": shard_log,
        }
        Path(out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

        if result.get("skipped"):
            note = "(skipped, already done)"
        else:
            note = f"{result['throughput_chunks_per_s']:.0f} ch/s"
        print(f"  [{i+1:>3}/{len(parquets)}] {p.name}  n={result['n']:>7,}  {note}  "
              f"total: {total_n:,}  rate: {rate:.0f} ch/s",
              flush=True)

    total_elapsed = time.monotonic() - total_start
    print(f"\nDone. {total_n:,} chunks in {total_elapsed/60:.1f} min "
          f"({total_n/total_elapsed:.0f} ch/s overall)")


if __name__ == "__main__":
    main()
