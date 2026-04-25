"""Split the monolithic ColBERT output `.npy` files into ≤ 2 GiB shards.

Each ColBERT dataset is produced by `embed_colbert_local.py` as a single
`data-00000-of-00001.npy` file (N_tokens × 64, fp16). When that file gets
large (~45 GB for our 1M-chunk runs) the SAE data loader must serve it
lazily to avoid OOM, which limits DataLoader `num_workers` to 0 and
slows multi-dataset training. Splitting to many ≤ 2 GiB shards removes
that constraint: each shard fits in the small-shard materialization
path with room to spare, and multiple workers can stream in parallel.

This script writes new `data-XXXXX-of-YYYYY.npy` files alongside the
original, then renames the original to `data-monolithic.npy.bak` so the
directory only lists properly-sharded files for the loader. Shards are
written as standard `np.save` headered .npy so the loader auto-detects
dtype.

chunk_offsets.npy is left as-is: the SAE training path does not use it,
and any chunk-aware downstream code computes global token index → shard
separately via the (cumulative) shard_sizes that the loader already
tracks.
"""
import argparse
import numpy as np
from pathlib import Path

# Leave a little headroom under the lazy-load threshold (2 GiB in the loader).
TARGET_BYTES = int(1.8 * 1024 ** 3)


def shard_one(path: Path, dry_run: bool = False):
    print(f"\n== {path.parent.name}/{path.name} ==")
    arr = np.load(path, mmap_mode="r")
    n_rows, dim = arr.shape
    itemsize = arr.dtype.itemsize
    total_bytes = n_rows * dim * itemsize
    rows_per_shard = TARGET_BYTES // (dim * itemsize)
    n_shards = (n_rows + rows_per_shard - 1) // rows_per_shard
    print(f"  shape {arr.shape} dtype {arr.dtype} total {total_bytes/1024**3:.2f} GiB")
    print(f"  → {n_shards} shards of up to {rows_per_shard:,} rows "
          f"({rows_per_shard*dim*itemsize/1024**3:.2f} GiB each)")
    if dry_run:
        return
    out_dir = path.parent
    # Write shards first
    for i in range(n_shards):
        s = i * rows_per_shard
        e = min(s + rows_per_shard, n_rows)
        out = out_dir / f"data-{i:05d}-of-{n_shards:05d}.npy"
        if out.exists():
            print(f"  skip {out.name} (exists)")
            continue
        sub = np.asarray(arr[s:e])            # materialize this slice (safe at ~2 GiB)
        np.save(out, sub)
        print(f"  wrote {out.name} ({e-s:,} rows)")
    # Then move the original out of the way so the loader won't pick it up
    # (loader restricts to data-*.npy; monolithic filename would still match).
    # Rename after writing so a crash mid-way leaves things safe.
    bak = out_dir / "data-monolithic.npy.bak"
    if path.exists() and not bak.exists():
        path.rename(bak)
        print(f"  renamed original -> {bak.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="Paths to the monolithic data-00000-of-00001.npy files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for p in args.paths:
        shard_one(Path(p), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
