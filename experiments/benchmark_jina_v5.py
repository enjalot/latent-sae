"""Throughput + memory benchmark for jina v5 nano / small.

Sweeps over (chunk_length × batch_size) and records:
  - chunks/sec
  - per-chunk latency (ms)
  - GPU memory used (max during run)

Tests against fineweb-edu chunked-500 corpus (chunks have actual lengths
~2000 chars, ~500 tokens). For target lengths > 500 tokens we
concatenate consecutive chunks to reach the target.

Usage:
  python -m experiments.benchmark_jina_v5 \\
      --models jinaai/jina-embeddings-v5-text-nano-retrieval,jinaai/jina-embeddings-v5-text-small-retrieval \\
      --slugs jina-v5-nano,jina-v5-small \\
      --lengths 500,1000,2000,4000 \\
      --batch-sizes 2,4,8,16,32 \\
      --n-chunks 256
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_test_corpus(target_token_lengths, n_per_length, tokenizer):
    """Build test docs of specific token lengths by tokenizing source chunks
    ONCE and concatenating token IDs (avoids O(N^2) re-tokenization)."""
    src = pd.read_parquet(
        "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train/data-00000-of-00099.parquet",
        columns=["chunk_text"]
    )["chunk_text"].head(min(20000, max(target_token_lengths) // 100 * n_per_length + 5000)).tolist()
    print(f"  pre-tokenizing {len(src)} source chunks...")
    t0 = time.monotonic()
    src_ids = [tokenizer.encode(s, add_special_tokens=False) for s in src]
    print(f"  done in {time.monotonic()-t0:.1f}s")

    out = {}
    for tlen in target_token_lengths:
        docs = []
        i = 0
        while len(docs) < n_per_length and i < len(src_ids):
            buf_ids = []
            while i < len(src_ids) and len(buf_ids) < tlen:
                buf_ids.extend(src_ids[i])
                i += 1
            if len(buf_ids) >= tlen:
                docs.append(tokenizer.decode(buf_ids[:tlen]))
        out[tlen] = docs
        print(f"  built {len(docs)} test docs at target {tlen} tokens "
              f"(mean actual chars: {int(np.mean([len(d) for d in docs])) if docs else 0})",
              flush=True)
    return out


def benchmark_one(model, docs, batch_size, device="cuda"):
    """Time encoding `docs` at batch_size, return (chunks_per_sec, mem_used_mib)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    n = len(docs)
    t0 = time.monotonic()
    try:
        vecs = model.encode(docs, batch_size=batch_size, show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True)
        elapsed = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return {"ok": True, "n": n, "elapsed_s": elapsed,
                 "chunks_per_sec": n / elapsed if elapsed > 0 else 0,
                 "ms_per_chunk": 1000 * elapsed / n if n > 0 else 0,
                 "peak_mem_mib": peak,
                 "vec_dim": vecs.shape[1]}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"ok": False, "error": "OOM", "n": n}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="jinaai/jina-embeddings-v5-text-nano-retrieval,jinaai/jina-embeddings-v5-text-small-retrieval")
    ap.add_argument("--slugs", default="jina-v5-nano,jina-v5-small")
    ap.add_argument("--lengths", default="500,1000,2000,4000")
    ap.add_argument("--batch-sizes", default="2,4,8,16,32,64")
    ap.add_argument("--n-chunks", type=int, default=256)
    ap.add_argument("--out", default="/data/jina_v5_benchmark.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    slugs = [s.strip() for s in args.slugs.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    from sentence_transformers import SentenceTransformer

    all_rows = []
    for model_name, slug in zip(models, slugs):
        print(f"\n=== {slug} ({model_name}) ===")
        model = SentenceTransformer(model_name,
                                     cache_folder="/data/hf/sentence-transformers",
                                     device=args.device, trust_remote_code=True)
        # max_seq_length is the model's default but we'll override per-test
        default_max = model.max_seq_length
        print(f"  default max_seq_length: {default_max}")

        # Build test corpora once per model (different tokenizers)
        print(f"  building test corpora at {lengths} tokens...")
        corpora = build_test_corpus(lengths, args.n_chunks, model.tokenizer)

        for tlen in lengths:
            if tlen > default_max:
                print(f"  skip tlen={tlen} (> default max_seq_length {default_max})")
                continue
            model.max_seq_length = tlen
            docs = corpora[tlen]
            if len(docs) < args.n_chunks:
                print(f"  warning: only {len(docs)} docs at tlen={tlen}")
            for bs in batch_sizes:
                # Warm-up (1 batch)
                _ = benchmark_one(model, docs[:bs], bs)
                # Real measurement
                row = benchmark_one(model, docs, bs)
                row.update({"slug": slug, "model": model_name,
                            "chunk_tokens": tlen, "batch_size": bs,
                            "vec_dim": row.get("vec_dim", -1)})
                all_rows.append(row)
                if row["ok"]:
                    print(f"    tlen={tlen:<5} bs={bs:<3}  {row['chunks_per_sec']:>7.1f} ch/s  "
                          f"{row['ms_per_chunk']:>7.2f} ms/ch  peak={row['peak_mem_mib']/1024:>5.2f} GB",
                          flush=True)
                else:
                    print(f"    tlen={tlen:<5} bs={bs:<3}  FAILED: {row['error']}",
                          flush=True)
                # Persist every step so timeouts don't lose data
                Path(args.out).write_text(json.dumps({"rows": all_rows}, indent=2))
        del model
        torch.cuda.empty_cache()
        gc.collect()

    Path(args.out).write_text(json.dumps({"rows": all_rows}, indent=2))
    print(f"\nwrote {args.out}")

    # Summary table
    df = pd.DataFrame([r for r in all_rows if r.get("ok")])
    if not df.empty:
        print("\n=== summary (chunks/sec) ===")
        pivot = df.pivot_table(index=["slug", "chunk_tokens"], columns="batch_size",
                                values="chunks_per_sec", aggfunc="mean")
        print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
