import os
import torch
import torch.distributed as dist
from contextlib import nullcontext, redirect_stdout
from simple_parsing import field, parse

from datasets import Dataset, load_dataset

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

    wandb_project: str = "sae-fw-nomic-text-v1.5"
    """Wandb project name."""

def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    # if not ddp or rank == 0:
    #     model, dataset = load_artifacts(args, rank)
    # if ddp:
    #     dist.barrier()
    #     if rank != 0:
    #         model, dataset = load_artifacts(args, rank)
    #     dataset = dataset.shard(dist.get_world_size(), rank)
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

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        print(f"Training on '{args.dataset}' (split '{args.split}')")
        # print(f"Storing model weights in {model.dtype}")

        trainer = SaeTrainer(args, dataset, device)
        trainer.fit()


if __name__ == "__main__":
    run()