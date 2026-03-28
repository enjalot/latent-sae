import os
from collections import OrderedDict
from typing import Union, List

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedEmbeddingDataset(Dataset):
    """Memory-efficient dataset for pre-computed embedding shards (.pt or .npy files).

    Uses an LRU cache to keep recently-accessed shards in memory.
    Supports loading from multiple directories for mixing data sources.
    """

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        cache_size: int = 10,
        d_in: int = 768,
        shuffle: bool = False,
        warm_up_cache: bool = True,
        file_type: str = "pt",
    ):
        self.file_type = file_type.lower()
        if self.file_type not in ("pt", "npy"):
            raise ValueError(f"Unsupported file_type: {self.file_type}. Use 'pt' or 'npy'.")

        self.data_dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        self.cache_size = cache_size
        self.d_in = d_in
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.shuffle = shuffle

        # Collect shard files from all directories
        self.shard_files: list[tuple[str, str]] = []
        for directory in self.data_dirs:
            files = sorted(f for f in os.listdir(directory) if f.endswith(f".{self.file_type}"))
            self.shard_files.extend((directory, f) for f in files)

        if not self.shard_files:
            raise FileNotFoundError(f"No .{self.file_type} files found in {self.data_dirs}")

        if shuffle:
            import random
            random.shuffle(self.shard_files)

        # Calculate sizes without loading data (assumes float32)
        self.shard_sizes = []
        for dir_path, f in self.shard_files:
            tensor_size = os.path.getsize(os.path.join(dir_path, f)) // (self.d_in * 4)
            self.shard_sizes.append(tensor_size)

        self.cumulative_sizes = np.cumsum(self.shard_sizes)
        self.total_size = int(self.cumulative_sizes[-1])
        print(f"ShardedEmbeddingDataset: {len(self.shard_files)} shards, {self.total_size:_} embeddings")

        if warm_up_cache:
            for shard_idx in range(min(cache_size, len(self.shard_files))):
                self._load_shard(shard_idx)

    def __len__(self) -> int:
        return self.total_size

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        dir_path, file_name = self.shard_files[shard_idx]
        file_path = os.path.join(dir_path, file_name)
        size = self.shard_sizes[shard_idx]

        if self.file_type == "pt":
            data = torch.load(file_path, weights_only=True)
        else:
            embeddings = np.memmap(file_path, dtype="float32", mode="r", shape=(size, self.d_in))
            data = torch.from_numpy(embeddings.copy())

        if self.shuffle:
            data = data[torch.randperm(data.shape[0])]

        # Store in LRU cache
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[shard_idx] = data
        self.cache.move_to_end(shard_idx)

        return data

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        idx_in_shard = idx - (int(self.cumulative_sizes[shard_idx - 1]) if shard_idx > 0 else 0)

        if shard_idx not in self.cache:
            self._load_shard(shard_idx)

        return self.cache[shard_idx][idx_in_shard]
