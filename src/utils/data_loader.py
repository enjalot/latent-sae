import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, load_from_disk
import pyarrow.parquet as pq
import numpy as np

class StreamingEmbeddingDataset(IterableDataset):
    def __init__(self, data_path, data_type, embedding_column, split="train"):
        self.data_path = data_path
        self.data_type = data_type
        self.embedding_column = embedding_column
        self.split = split

    def __iter__(self):
        if self.data_type == 'huggingface':
            # dataset = load_dataset("arrow", data_dir=self.data_path, streaming=True)
            dataset = load_from_disk(self.data_path)
            try:
                for item in dataset[self.split]:
                    yield torch.tensor(item[self.embedding_column], dtype=torch.float32)
            except:
                # TODO: why does the last row in a dataset seem to screw up?
                # I saved the dataset with datasets.save_to_disk()
                pass
        elif self.data_type == 'parquet':
            parquet_file = pq.ParquetFile(self.data_path)
            for batch in parquet_file.iter_batches():
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    yield torch.tensor(row[self.embedding_column], dtype=torch.float32)

    def __len__(self):
        if self.data_type == 'huggingface':
            dataset = load_from_disk(self.data_path)
            nr = dataset[self.split].num_rows
            return nr
        elif self.data_type == 'parquet':
            parquet_file = pq.ParquetFile(self.data_path)
            return parquet_file.metadata.num_rows
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

def get_streaming_dataloader(data_path, data_type, embedding_column, batch_size=32, split="train"):
    streaming_dataset = StreamingEmbeddingDataset(data_path, data_type, embedding_column, split)
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