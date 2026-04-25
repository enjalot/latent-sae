import os
from collections import OrderedDict
from typing import Union, List

import numpy as np
import torch
from torch.utils.data import Dataset


_NPY_MAGIC = b"\x93NUMPY"

# Shards larger than this get served as lazy memmaps (one row faulted per
# __getitem__) rather than materialized into RAM. Kept intentionally low:
# the in-RAM cache stores fp32 copies, which is 2x the on-disk fp16 size,
# and PyTorch DataLoader's forked workers each build their own LRU cache,
# multiplying the anonymous-memory footprint by num_workers. Lazy mmap is
# file-backed and pages shared by the OS across workers, so RSS stays flat.
_LARGE_SHARD_BYTES = 256 * 1024 ** 2   # 256 MiB — anything larger goes lazy


def _looks_like_npy(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(6) == _NPY_MAGIC


_DTYPE_ITEMSIZE = {
    "float32": 4, "float16": 2, "bfloat16": 2,
}


class ShardedEmbeddingDataset(Dataset):
    """Memory-efficient dataset for pre-computed embedding shards (.pt or .npy files).

    Supports two on-disk ``.npy`` formats:
      * Proper numpy ``.npy`` files (with the \\x93NUMPY header) — dtype/shape
        are read from the header. Produced by ``np.save``.
      * Raw memmap dumps (no header) — rows are packed float values written
        directly to disk. Used by the MiniLM embeddings on this workstation.
        For these, pass ``raw_dtype`` to specify the element dtype.

    Memory strategy:
      * Small shards (< _LARGE_SHARD_BYTES) are materialized into an LRU
        in-RAM tensor cache on first access (fast random reads).
      * Large shards — e.g. the 47 GB per-token ColBERT outputs — are kept as
        lazy ``np.memmap`` objects and converted to ``float32`` **per row** in
        ``__getitem__``. The OS page cache handles working-set residency; RSS
        stays bounded and mmap pages are shared across DataLoader workers via
        fork + COW. Materializing a shard this size would OOM an 128 GB box
        (fp16 → fp32 roughly doubles it, times N workers).
    """

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        cache_size: int = 10,
        d_in: int = 768,
        shuffle: bool = False,
        warm_up_cache: bool = True,
        file_type: str = "pt",
        raw_dtype: str = "float32",
    ):
        self.file_type = file_type.lower()
        if self.file_type not in ("pt", "npy"):
            raise ValueError(f"Unsupported file_type: {self.file_type}. Use 'pt' or 'npy'.")

        self.data_dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        self.cache_size = cache_size
        self.d_in = d_in
        # Cache stores tensors only for small shards; large-shard lazy memmaps
        # live in self.lazy_arrays.
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.lazy_arrays: dict[int, np.ndarray] = {}
        self.shuffle = shuffle
        self.raw_dtype = raw_dtype
        if raw_dtype not in _DTYPE_ITEMSIZE:
            raise ValueError(f"Unsupported raw_dtype: {raw_dtype}. Use {list(_DTYPE_ITEMSIZE)}.")

        # Collect shard files. When the convention includes sidecar metadata
        # (e.g. chunk_offsets.npy), restrict to actual embedding shards named
        # data-*.npy when that pattern is present.
        self.shard_files: list[tuple[str, str]] = []
        for directory in self.data_dirs:
            files = sorted(f for f in os.listdir(directory) if f.endswith(f".{self.file_type}"))
            data_files = [f for f in files if f.startswith("data-") or f.startswith("data_")]
            if data_files:
                files = data_files
            self.shard_files.extend((directory, f) for f in files)

        if not self.shard_files:
            raise FileNotFoundError(f"No .{self.file_type} files found in {self.data_dirs}")

        if shuffle:
            import random
            random.shuffle(self.shard_files)

        # Per-shard metadata — size in rows, format flag, and "is lazy" flag.
        self.shard_sizes: list[int] = []
        self.shard_is_npy: list[bool] = []
        self.shard_is_lazy: list[bool] = []
        n_lazy = 0
        for dir_path, f in self.shard_files:
            path = os.path.join(dir_path, f)
            on_disk = os.path.getsize(path)
            if self.file_type == "npy" and _looks_like_npy(path):
                arr = np.load(path, mmap_mode="r")
                assert arr.shape[1] == self.d_in, (
                    f"{path}: d_in mismatch ({arr.shape[1]} vs expected {self.d_in})")
                self.shard_sizes.append(arr.shape[0])
                self.shard_is_npy.append(True)
            else:
                itemsize = _DTYPE_ITEMSIZE[self.raw_dtype]
                self.shard_sizes.append(on_disk // (self.d_in * itemsize))
                self.shard_is_npy.append(False)
            is_lazy = on_disk >= _LARGE_SHARD_BYTES
            self.shard_is_lazy.append(is_lazy)
            n_lazy += int(is_lazy)

        self.cumulative_sizes = np.cumsum(self.shard_sizes)
        self.total_size = int(self.cumulative_sizes[-1])
        print(f"ShardedEmbeddingDataset: {len(self.shard_files)} shards, "
              f"{self.total_size:_} embeddings, {n_lazy} lazy (memmap)")

        if warm_up_cache:
            # Only warm up small shards — touching a 47 GB memmap here would
            # fault in the whole file.
            n_warmed = 0
            for shard_idx in range(len(self.shard_files)):
                if self.shard_is_lazy[shard_idx]:
                    continue
                self._load_shard(shard_idx)
                n_warmed += 1
                if n_warmed >= cache_size:
                    break

    def __len__(self) -> int:
        return self.total_size

    def _open_memmap(self, shard_idx: int) -> np.ndarray:
        """Return a lazy np.memmap view of the shard (no copy, dtype unchanged)."""
        if shard_idx in self.lazy_arrays:
            return self.lazy_arrays[shard_idx]
        dir_path, file_name = self.shard_files[shard_idx]
        path = os.path.join(dir_path, file_name)
        if self.shard_is_npy[shard_idx]:
            arr = np.load(path, mmap_mode="r")
        else:
            arr = np.memmap(path, dtype=self.raw_dtype, mode="r",
                            shape=(self.shard_sizes[shard_idx], self.d_in))
        self.lazy_arrays[shard_idx] = arr
        return arr

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        """Materialize a SMALL shard into an in-RAM float32 tensor + LRU cache."""
        dir_path, file_name = self.shard_files[shard_idx]
        file_path = os.path.join(dir_path, file_name)

        if self.file_type == "pt":
            data = torch.load(file_path, weights_only=True)
        elif self.shard_is_npy[shard_idx]:
            arr = np.load(file_path, mmap_mode="r")
            data = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        else:
            size = self.shard_sizes[shard_idx]
            embeddings = np.memmap(file_path, dtype=self.raw_dtype, mode="r",
                                   shape=(size, self.d_in))
            data = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))

        if self.shuffle:
            data = data[torch.randperm(data.shape[0])]

        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[shard_idx] = data
        self.cache.move_to_end(shard_idx)
        return data

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        idx_in_shard = idx - (int(self.cumulative_sizes[shard_idx - 1]) if shard_idx > 0 else 0)

        if self.shard_is_lazy[shard_idx]:
            arr = self._open_memmap(shard_idx)
            row = arr[idx_in_shard]
            # Small per-row copy + dtype cast. Pages get faulted in lazily
            # by the OS and shared across fork()ed workers via the page cache.
            return torch.from_numpy(np.asarray(row, dtype=np.float32))

        if shard_idx not in self.cache:
            self._load_shard(shard_idx)
        return self.cache[shard_idx][idx_in_shard]
