"""
Autoresearch training script — AGENT-EDITABLE.

This is the file the autoresearch agent modifies. It defines the SAE
architecture, hyperparameters, and training procedure.

The agent should edit this file to test ideas, then run it.
prepare.py (evaluation) is locked and cannot be edited.

Usage:
  python experiments/autoresearch/train.py \
    --data-dir /path/to/embedding/shards \
    --d-in 384 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --device cpu

Output (grep-friendly):
  ---
  composite_score: 0.747000
  clinc150_sparse: 0.796000
  ...
"""

import argparse
import sys
import time
import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from latentsae.sae import Sae
from latentsae.utils.config import SaeConfig, SaeType, TrainConfig, LRSchedule
from latentsae.trainer import SaeTrainer
from latentsae.utils.data_loader import ShardedEmbeddingDataset
from experiments.autoresearch.prepare import evaluate, format_results

# ════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to test ideas
# ════════════════════════════════════════════════════════════════

SAE_CONFIG = dict(
    sae_type="topk",
    expansion_factor=4,
    k=128,
    normalize_decoder=True,
    # Auxiliary loss for dead feature revival
    # k_anneal=False,
    # multi_topk=False,
)

TRAIN_CONFIG = dict(
    batch_size=1024,
    grad_acc_steps=1,
    lr=None,  # Auto-scaled
    lr_schedule="cosine",
    lr_warmup_steps=200,
    use_amp=True,
    auxk_alpha=0.03125,  # 1/32
    dead_feature_threshold=50000,
    save_every=999999,  # Don't checkpoint during autoresearch
)

N_SAMPLES = 2_000_000  # 2M samples for ~20s training
TIME_BUDGET = 300  # 5 minutes total (training + eval)

# ════════════════════════════════════════════════════════════════


class SyntheticEmbeddingDataset(Dataset):
    """Fallback dataset of random unit vectors for local testing."""
    def __init__(self, n, d):
        self.data = torch.randn(n, d)
        self.data = self.data / self.data.norm(dim=1, keepdim=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", nargs="+", default=None,
                        help="Embedding shard directories. If not provided, uses synthetic data.")
    parser.add_argument("--d-in", type=int, default=384)
    parser.add_argument("--file-type", default="npy")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    t_start = time.time()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    dev = torch.device(device)

    # Dataset
    if args.data_dir:
        dataset = ShardedEmbeddingDataset(
            args.data_dir, cache_size=10, d_in=args.d_in,
            shuffle=True, warm_up_cache=False, file_type=args.file_type,
        )
        if N_SAMPLES < len(dataset):
            dataset = Subset(dataset, list(range(N_SAMPLES)))
    else:
        print("No data dir provided, using synthetic data", file=sys.stderr)
        dataset = SyntheticEmbeddingDataset(N_SAMPLES, args.d_in)

    # Build configs
    sae_cfg = SaeConfig(**{k: (SaeType(v) if k == "sae_type" else v)
                           for k, v in SAE_CONFIG.items()})
    train_cfg = TrainConfig(
        sae=sae_cfg, d_in=args.d_in,
        log_to_wandb=False,
        checkpoints_directory="/tmp/autoresearch_ckpt",
        **{k: (LRSchedule(v) if k == "lr_schedule" else v)
           for k, v in TRAIN_CONFIG.items()},
    )

    # Train
    print(f"Training: {len(dataset):_} samples, {sae_cfg.sae_type.value} "
          f"{sae_cfg.expansion_factor}x k={sae_cfg.k}", file=sys.stderr)

    trainer = SaeTrainer(train_cfg, dataset, dev)
    trainer.fit()
    train_time = time.time() - t_start

    # Evaluate
    print(f"Training done in {train_time:.1f}s, evaluating...", file=sys.stderr)
    sae = trainer.sae
    results = evaluate(sae, args.embedding_model, device=device)
    results["training_time_s"] = train_time
    results["total_time_s"] = time.time() - t_start

    # Output — structured for grep
    print(format_results(results))


if __name__ == "__main__":
    main()
