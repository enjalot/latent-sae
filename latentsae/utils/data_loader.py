import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
from datasets import load_dataset, load_from_disk
import pyarrow.parquet as pq
import numpy as np
import random
import time
import os
from collections import OrderedDict

class ShardedEmbeddingDataset(Dataset):
    def __init__(self, data_dir, cache_size=10, d_in=768, shuffle=False, warm_up_cache=True, file_type='pt'):
        # Add file_type parameter
        self.file_type = file_type.lower()
        if self.file_type not in ['pt', 'npy']:
            raise ValueError(f"Unsupported file_type: {self.file_type}")

        # Convert single data_dir to list for consistent handling
        self.data_dirs = [data_dir] if isinstance(data_dir, str) else data_dir
        print("data dirs", self.data_dirs)
        self.cache_size = cache_size
        self.d_in = d_in
        self.cache = OrderedDict()
        self.shuffle = shuffle

        # Collect files from all directories
        self.shard_files = []
        self.dir_indices = []  # Keep track of which directory each file belongs to
        for dir_idx, directory in enumerate(self.data_dirs):
            files = sorted([f for f in os.listdir(directory) if f.endswith(f'.{self.file_type}')])
            self.shard_files.extend([(directory, f) for f in files])
            self.dir_indices.extend([dir_idx] * len(files))

        print("shard files", len(self.shard_files))
        # After collecting files but before calculating sizes, shuffle the shard files if needed
        if shuffle:
            combined = list(zip(self.shard_files, self.dir_indices))
            random.shuffle(combined)
            self.shard_files, self.dir_indices = zip(*combined)
            # self.shard_files = list(self.shard_files)
            # self.dir_indices = list(self.dir_indices)

        # Calculate sizes without loading data
        self.shard_sizes = []
        for dir_path, f in self.shard_files:
            tensor_size = os.path.getsize(os.path.join(dir_path, f)) // (self.d_in * 4)
            self.shard_sizes.append(tensor_size)
        
        self.cumulative_sizes = np.cumsum(self.shard_sizes)
        self.total_size = self.cumulative_sizes[-1]
        if warm_up_cache:
            self.warm_up_cache(self.cache_size)

    def warm_up_cache(self, num_shards=10):
        """Preload all shards into the cache."""
        for shard_idx in range(num_shards):
            self.load_shard(shard_idx)

    def __len__(self):
        return self.total_size

    def load_shard(self, shard_idx):
        dir_path, file_name = self.shard_files[shard_idx]
        file_path = os.path.join(dir_path, file_name)
        size = self.shard_sizes[shard_idx]
        
        if self.file_type == 'pt':
            data = torch.load(file_path)
        else:  # npy
            print("loading npy", file_path)
            embeddings = np.memmap(file_path, 
                      dtype='float32', 
                      mode='r', 
                      shape=(size, self.d_in))
            data = torch.from_numpy(embeddings.copy())  # Copy to ensure it's writable if we need to shuffle
            
        if self.shuffle:
            data = data[torch.randperm(data.shape[0])]
        return data

    def __getitem__(self, idx):
        shard_idx = np.searchsorted(self.cumulative_sizes, idx + 1)
        if shard_idx > 0:
            idx_in_shard = idx - self.cumulative_sizes[shard_idx - 1]
        else:
            idx_in_shard = idx

        if shard_idx not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[shard_idx] = self.load_shard(shard_idx)
            self.cache.move_to_end(shard_idx)

        return self.cache[shard_idx][idx_in_shard]

# Usage
# dataset = ShardedDataset("path/to/preprocessed_data", cache_size=10, d_in=768)
# dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

"""
This is incredibly slow, at least 60x slower
"""
class StreamingEmbeddingDataset(IterableDataset):
    def __init__(self, data_path, data_type, embedding_column, split="train", buffer_size=500000):
        self.data_path = data_path
        self.data_type = data_type
        self.embedding_column = embedding_column
        self.split = split
        self.dataset = None
        self.buffer_size = buffer_size

    def __iter__(self):
        if self.data_type == 'huggingface':
            # dataset = load_dataset("arrow", data_dir=self.data_path, streaming=True)
            print("loading for iteration", self.data_path)
            if self.dataset is None:
                self.dataset = load_from_disk(self.data_path, keep_in_memory=False)
            print("loaded")
            for item in self.dataset[self.split]:
                yield torch.tensor(item[self.embedding_column], dtype=torch.float32)
        elif self.data_type == 'parquet':
            if self.dataset is None:
                self.dataset = pq.ParquetFile(self.data_path)
            for batch in self.dataset.iter_batches():
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    yield torch.tensor(row[self.embedding_column], dtype=torch.float32)

    def __len__(self):
        if self.data_type == 'huggingface':
            print("loading", self.data_path)
            if self.dataset is None:
                self.dataset = load_from_disk(self.data_path, keep_in_memory=False)
            nr = self.dataset[self.split].num_rows
            print("loaded", nr)
            return nr
        elif self.data_type == 'parquet':
            if self.dataset is None:
                self.dataset = pq.ParquetFile(self.data_path)
            return self.dataset.metadata.num_rows
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

