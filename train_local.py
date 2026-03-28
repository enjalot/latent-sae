"""
Train an SAE locally on pre-computed embeddings.

Examples:
  # TopK SAE (default)
  python train_local.py --batch_size=512 --grad_acc_steps=4 --k=64 --expansion_factor=32

  # Gated SAE
  python train_local.py --batch_size=512 --sae_type=gated --k=64

  # JumpReLU SAE
  python train_local.py --batch_size=512 --sae_type=jumprelu
"""
import torch
from simple_parsing import parse

from latentsae.trainer import SaeTrainer, TrainConfig
from latentsae.utils.data_loader import ShardedEmbeddingDataset


class RunConfig(TrainConfig):
    data_dir: str = "./notebooks/data/test_train"
    """Path to sharded embedding data directory (or comma-separated list of directories)."""

    file_type: str = "pt"
    """Shard file format: 'pt' (torch) or 'npy' (numpy memmap)."""

    shuffle: bool = True
    """Shuffle data within shards."""

    wandb_project: str = "latent-sae-local"
    """Wandb project name."""


def run():
    args = parse(RunConfig)

    data_dirs = [d.strip() for d in args.data_dir.split(",")]
    if len(data_dirs) == 1:
        data_dirs = data_dirs[0]

    dataset = ShardedEmbeddingDataset(
        data_dirs,
        cache_size=10,
        d_in=args.d_in,
        shuffle=args.shuffle,
        warm_up_cache=True,
        file_type=args.file_type,
    )
    print(f"Dataset: {len(dataset):_} embeddings")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    trainer = SaeTrainer(args, dataset, device)
    trainer.fit()


if __name__ == "__main__":
    run()
