"""OTF throughput + memory probe.

Drains OnTheFlyColBERTDataset at max speed for a fixed wall time and
reports steady-state tok/s, queue depth, anon RSS delta, GPU memory
delta. Used to pick chunks_per_encode / queue_size for Phase 16.

Run a single config:
    python experiments/bench_otf.py --chunks-per-encode 128 --queue-size 4 --duration 30

Run the sweep defined below:
    python experiments/bench_otf.py --sweep
"""
import argparse
import gc
import os
import resource
import time

import torch

from latentsae.utils.otf_dataset import OnTheFlyColBERTDataset

PARQUET_DIRS = [
    "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
    "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-500/train",
    "/data/chunks/pile-uncopyrighted-chunked-500/train",
]


def _anon_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024
    return -1.0


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return -1.0
    return torch.cuda.memory_allocated() / 1024 ** 2


def bench(chunks_per_encode: int, queue_size: int, duration: float,
          batch_size: int = 8192, warmup: float = 8.0, drain_only: bool = True) -> dict:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    anon0 = _anon_rss_mb()
    gpu0 = _gpu_mem_mb()

    ds = OnTheFlyColBERTDataset(
        parquet_dirs=PARQUET_DIRS,
        batch_size=batch_size,
        chunks_per_encode=chunks_per_encode,
        queue_size=queue_size,
        device="cuda",
        seed=42,
    )
    it = iter(ds)

    # Warmup (drain but don't count)
    t_warm_end = time.monotonic() + warmup
    n_warm = 0
    while time.monotonic() < t_warm_end:
        try:
            _ = next(it)
            n_warm += 1
        except StopIteration:
            break

    # Steady state
    tok_at_warm = ds._tokens_produced
    t0 = time.monotonic()
    t_end = t0 + duration
    n_batches = 0
    queue_samples = []
    while time.monotonic() < t_end:
        try:
            batch = next(it)
        except StopIteration:
            break
        if not drain_only:
            # Simulate minimal SAE touch (copy to GPU)
            batch = batch.to("cuda", non_blocking=True)
        n_batches += 1
        if n_batches % 50 == 0:
            queue_samples.append(ds._q.qsize() if ds._q is not None else -1)
    t_wall = time.monotonic() - t0
    tok_made = ds._tokens_produced - tok_at_warm

    peak_gpu = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else -1.0
    anon1 = _anon_rss_mb()
    gpu1 = _gpu_mem_mb()

    ds.stop()
    del ds, it
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "chunks_per_encode": chunks_per_encode,
        "queue_size": queue_size,
        "batch_size": batch_size,
        "duration_s": round(t_wall, 2),
        "n_sae_batches": n_batches,
        "tokens_s": round(tok_made / t_wall, 0),
        "sae_batches_s": round(n_batches / t_wall, 2),
        "queue_depth_avg": round(sum(queue_samples) / len(queue_samples), 2) if queue_samples else -1,
        "queue_depth_max": max(queue_samples) if queue_samples else -1,
        "anon_rss_mb_start": round(anon0, 0),
        "anon_rss_mb_end": round(anon1, 0),
        "anon_rss_delta_mb": round(anon1 - anon0, 0),
        "gpu_mb_peak": round(peak_gpu, 0),
        "gpu_mb_end": round(gpu1, 0),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-per-encode", type=int, default=64)
    ap.add_argument("--queue-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        configs = [
            (32, 4),
            (64, 4),
            (128, 4),
            (256, 4),
            (128, 8),
            (256, 8),
        ]
        results = []
        for cpe, qs in configs:
            print(f"\n--- chunks_per_encode={cpe} queue_size={qs} ---", flush=True)
            r = bench(cpe, qs, duration=args.duration, warmup=args.warmup,
                      batch_size=args.batch_size)
            results.append(r)
            print(r, flush=True)
        print("\n=== SUMMARY ===")
        print(f"{'cpe':>4} {'qs':>3} {'tok/s':>10} {'batch/s':>7} {'q_avg':>6} {'q_max':>5} {'anon_Δ':>7} {'gpu_peak':>9}")
        for r in results:
            print(f"{r['chunks_per_encode']:>4} {r['queue_size']:>3} {r['tokens_s']:>10,.0f} "
                  f"{r['sae_batches_s']:>7.2f} {r['queue_depth_avg']:>6.2f} "
                  f"{r['queue_depth_max']:>5d} {r['anon_rss_delta_mb']:>7.0f} {r['gpu_mb_peak']:>9.0f}")
    else:
        r = bench(args.chunks_per_encode, args.queue_size,
                  duration=args.duration, warmup=args.warmup,
                  batch_size=args.batch_size)
        print(r)


if __name__ == "__main__":
    main()
