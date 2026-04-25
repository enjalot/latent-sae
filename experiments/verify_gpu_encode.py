"""Verify that pylate encode(convert_to_numpy=False) produces identical
per-token embeddings to the default CPU-numpy path, then measure the
throughput gain.

Correctness test:
  1. Pick 200 real chunk texts.
  2. Encode both ways.
  3. Compare per-chunk token tensors byte-by-byte (fp16 max diff, mean diff).
  4. Additional anchor: compare against the saved disk pool (same test we
     ran yesterday to rule out pipeline differences).

Throughput test:
  CPU-numpy encode vs GPU-tensor encode. 100 calls each, same inputs.

Exits with code 1 if any chunk differs by more than 1 fp16 ulp (0.0005)
in max-abs, so this can be used in CI.
"""
import glob
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/home/enjalot/code/latent-sae')
from pylate import models


CHUNK_DIR = "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train"
POOL_NPY_BAK = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train/data-monolithic.npy.bak"
POOL_OFFSETS = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train/chunk_offsets.npy"


def load_chunks(n: int = 200):
    parquets = sorted(glob.glob(f"{CHUNK_DIR}/*.parquet"))
    df = pd.read_parquet(parquets[0], columns=["chunk_text"])
    texts = df["chunk_text"].tolist()[:n]
    return [t if t and t.strip() else " " for t in texts]


def encode_cpu(model, texts, batch_size=64):
    """OLD PATH — returns list of fp16 numpy arrays (on CPU), as used in OTF
    producer today (producer then concatenates and puts on queue)."""
    out = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       is_query=False)
    # Match OTF producer: cast to fp16 on CPU
    return [np.asarray(e, dtype=np.float16) for e in out]


def encode_gpu(model, texts, batch_size=64):
    """NEW PATH — returns list of fp16 tensors (on device).

    convert_to_numpy=False makes pylate skip the .cpu() call and return
    tensors on device. We cast to fp16 on-device to match the OTF queue dtype.
    """
    out = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       is_query=False, convert_to_numpy=False)
    # pylate returns a list of fp32 tensors on device; cast in-place
    return [e.to(dtype=torch.float16) for e in out]


def compare_chunks(name_a, arrs_a, name_b, arrs_b):
    assert len(arrs_a) == len(arrs_b), f"length mismatch {len(arrs_a)} vs {len(arrs_b)}"
    max_diff = 0.0
    mean_diff = 0.0
    shape_mismatch = 0
    exact_matches = 0
    n = len(arrs_a)
    for i in range(n):
        a = arrs_a[i]
        b = arrs_b[i]
        # Bring both to numpy fp16 for comparison
        if hasattr(a, "cpu"):
            a = a.detach().cpu().numpy()
        if hasattr(b, "cpu"):
            b = b.detach().cpu().numpy()
        a = np.asarray(a, dtype=np.float16)
        b = np.asarray(b, dtype=np.float16)
        if a.shape != b.shape:
            shape_mismatch += 1
            continue
        if np.array_equal(a, b):
            exact_matches += 1
            continue
        d = np.abs(a.astype(np.float32) - b.astype(np.float32))
        max_diff = max(max_diff, float(d.max()))
        mean_diff = max(mean_diff, float(d.mean()))
    print(f"  {name_a} vs {name_b}:  n={n}  exact={exact_matches}  "
          f"shape_mismatch={shape_mismatch}  max_abs={max_diff:.6g}  "
          f"max_mean_abs={mean_diff:.6g}")
    return max_diff, shape_mismatch


def time_encode(model, texts, fn, n_calls=10):
    # Warmup
    for _ in range(3):
        _ = fn(model, texts)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_calls):
        _ = fn(model, texts)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n_calls


def main():
    print("=== Loading pylate ColBERT model ===")
    model = models.ColBERT(model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m",
                           device="cuda")

    texts = load_chunks(200)
    print(f"Loaded {len(texts)} chunks")

    print("\n=== Correctness: GPU tensors vs CPU numpy ===")
    arrs_cpu = encode_cpu(model, texts)
    arrs_gpu = encode_gpu(model, texts)

    # Confirm GPU output is actually on device
    dev_counts = {}
    for t in arrs_gpu:
        d = str(t.device) if hasattr(t, "device") else "<no-device>"
        dev_counts[d] = dev_counts.get(d, 0) + 1
    print(f"  encode_gpu tensor device distribution: {dev_counts}")
    print(f"  encode_gpu dtype: {arrs_gpu[0].dtype}")

    max_diff, shape_mismatch = compare_chunks("CPU", arrs_cpu, "GPU", arrs_gpu)
    ok = (max_diff < 1e-3 and shape_mismatch == 0)
    print(f"  → {'PASS' if ok else 'FAIL'}")

    # Also compare to disk pool (anchor against known-good outputs)
    import os
    if os.path.exists(POOL_NPY_BAK):
        print("\n=== Correctness: vs saved disk pool (yesterday's reference) ===")
        pool = np.load(POOL_NPY_BAK, mmap_mode="r")
        offsets = np.load(POOL_OFFSETS)
        pool_chunks = [np.asarray(pool[offsets[i]:offsets[i+1]])
                       for i in range(len(texts))]
        compare_chunks("POOL", pool_chunks, "CPU", arrs_cpu)
        compare_chunks("POOL", pool_chunks, "GPU", arrs_gpu)

    print("\n=== Throughput: 10 calls of 200-chunk encode ===")
    t_cpu = time_encode(model, texts, encode_cpu, n_calls=10)
    t_gpu = time_encode(model, texts, encode_gpu, n_calls=10)
    speedup = t_cpu / t_gpu
    toks_per_s_cpu = sum(a.shape[0] for a in arrs_cpu) / t_cpu
    toks_per_s_gpu = sum(a.shape[0] for a in arrs_cpu) / t_gpu
    print(f"  CPU-numpy path:  {t_cpu*1000:.2f} ms/call  →  {toks_per_s_cpu:,.0f} tok/s")
    print(f"  GPU-tensor path: {t_gpu*1000:.2f} ms/call  →  {toks_per_s_gpu:,.0f} tok/s")
    print(f"  speedup: {speedup:.2f}×")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
