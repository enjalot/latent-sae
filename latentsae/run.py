import os
import torch
from simple_parsing import field, parse

from trainer import SaeTrainer, TrainConfig
from utils.data_loader import StreamingEmbeddingDataset

class RunConfig(TrainConfig):
    dataset: str = "./notebooks/data/test_train2"
    """Path to the dataset to use for training."""

    data_type: str = "huggingface"
    """Type of the dataset to use for training. 
    current options: huggingface or parquet"""

    split: str = "train"
    """Dataset split to use for training."""

    embedding_column: str = "embedding"
    """Column to use for embedding."""

    wandb_project: str = "sae-fw-modal-nomic-text-v1.5"
    """Wandb project name."""

def run():
    args = parse(RunConfig)
    print("loading dataset", args.dataset)
    dataset = StreamingEmbeddingDataset(args.dataset, args.data_type, args.embedding_column, split=args.split)
    print("LEN OF THE DATASET", len(dataset))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Training on '{args.dataset}' (split '{args.split}')")
    trainer = SaeTrainer(args, dataset, device)
    trainer.fit()


if __name__ == "__main__":
    run()