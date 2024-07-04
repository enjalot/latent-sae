import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_from_disk
import pyarrow.parquet as pq
import numpy as np

class StreamingEmbeddingDataset(IterableDataset):
    def __init__(self, dataset, embedding_column):
        self.dataset = dataset
        self.embedding_column = embedding_column

    def __iter__(self):
        for item in self.dataset:
            yield torch.tensor(item[self.embedding_column], dtype=torch.float32)
            
    def __len__(self):
        return len(self.dataset)

def get_streaming_dataloader(data_path, data_type, embedding_column, batch_size=32):
    if data_type == 'huggingface':
        dataset = load_from_disk(data_path)
    elif data_type == 'parquet':
        dataset = pq.ParquetDataset(data_path)
    else:
        raise ValueError("Invalid data_type. Choose 'huggingface' or 'parquet'.")

    streaming_dataset = StreamingEmbeddingDataset(dataset, embedding_column)
    return DataLoader(streaming_dataset, batch_size=batch_size)

class DummyEmbeddingDataset(IterableDataset):
    def __init__(self, num_samples, embed_dim):
        self.num_samples = num_samples
        self.embed_dim = embed_dim

    def __iter__(self):
        for _ in range(self.num_samples):
            yield torch.tensor(np.random.randn(self.embed_dim).astype(np.float32))

    def __len__(self):
        return self.num_samples

def get_dummy_dataloader(num_samples, embed_dim, batch_size=32):
    dummy_dataset = DummyEmbeddingDataset(num_samples, embed_dim)
    return DataLoader(dummy_dataset, batch_size=batch_size)