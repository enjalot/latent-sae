"""
Run SAE training experiments from YAML configs.

Usage:
  # Single experiment (local)
  python -m experiments.run_experiment experiments/configs/smoke_test.yaml

  # With CLI overrides
  python -m experiments.run_experiment experiments/configs/gpu_bench_base.yaml \
    --override sae.k=128 --override train.batch_size=1024

  # Parameter sweep
  python -m experiments.run_experiment experiments/configs/arch_sweep_base.yaml \
    --sweep experiments/configs/arch_sweep_type.yaml

  # Dry run (show what would run)
  python -m experiments.run_experiment experiments/configs/arch_sweep_base.yaml \
    --sweep experiments/configs/arch_sweep_type.yaml --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from .experiment_config import (
    ExperimentConfig,
    generate_sweep_configs,
    load_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_completed_runs(results_dir: str) -> dict:
    """Scan results_dir for completed runs, return {config_hash: results_dict}."""
    completed = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return completed
    for run_dir in results_path.iterdir():
        results_file = run_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                h = results.get("config_hash", "")
                if h:
                    completed[h] = results
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_single_experiment(config: ExperimentConfig, device: str = "auto") -> dict:
    """Run one SAE training experiment. Returns results dict."""
    import torch
    from latentsae.trainer import SaeTrainer, GPU_HOURLY_RATES
    from latentsae.utils.data_loader import ShardedEmbeddingDataset

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    # Create run directory
    run_dir = config.run_dir()
    os.makedirs(run_dir, exist_ok=True)
    config.to_yaml(os.path.join(run_dir, "config.yaml"))

    # Load data
    logger.info(f"Loading data from {len(config.data.datasets)} source(s)...")
    dataset = ShardedEmbeddingDataset(
        config.data.datasets,
        cache_size=10,
        d_in=config.data.d_in,
        shuffle=config.data.shuffle,
        warm_up_cache=False,
        file_type=config.data.file_type,
        raw_dtype=config.data.raw_dtype,
    )

    # Subsample if requested (for benchmarks)
    if config.data.n_samples and config.data.n_samples < len(dataset):
        from torch.utils.data import Subset
        indices = list(range(config.data.n_samples))
        dataset = Subset(dataset, indices)
        logger.info(f"Subsampled to {config.data.n_samples:_} samples")
    else:
        logger.info(f"Using all {len(dataset):_} samples")

    # Build TrainConfig and set checkpoints to run_dir
    train_config = config.to_train_config()
    if config.logging.save_model:
        train_config.checkpoints_directory = os.path.join(run_dir, "checkpoints")
        os.makedirs(train_config.checkpoints_directory, exist_ok=True)
    else:
        train_config.save_every = 999_999_999  # effectively disable checkpointing
        train_config.checkpoints_directory = os.path.join(run_dir, "checkpoints")
        os.makedirs(train_config.checkpoints_directory, exist_ok=True)

    # Update wandb run name to include experiment name
    if config.logging.use_wandb and not config.logging.wandb_run_name:
        train_config.run_name = config.name

    # Train
    logger.info(f"Starting experiment: {config.name} (hash: {config.config_hash()})")
    logger.info(f"Device: {dev}, GPU type: {config.infra.gpu_type}")

    t0 = time.time()
    trainer = SaeTrainer(train_config, dataset, dev)
    trainer.fit()
    total_time = time.time() - t0

    # Compile results
    n_params = sum(p.numel() for p in trainer.sae.parameters())
    results = {
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {
            "n_samples": len(dataset),
            "d_in": config.data.d_in,
            "n_sources": len(config.data.datasets),
        },
        "model": {
            "sae_type": config.sae.get("sae_type", "topk"),
            "expansion_factor": config.sae.get("expansion_factor", 32),
            "k": config.sae.get("k", 64),
            "n_params": n_params,
        },
        "timing": {
            "total_training_time_s": total_time,
            "avg_throughput_samples_sec": len(dataset) / max(total_time, 1),
        },
        "infra": {
            "gpu_type": config.infra.gpu_type,
            "device": str(dev),
        },
    }

    # Cost estimate
    if config.infra.gpu_type in GPU_HOURLY_RATES:
        results["timing"]["cost_estimate_usd"] = (
            total_time * GPU_HOURLY_RATES[config.infra.gpu_type] / 3600
        )

    # GPU memory
    if dev.type == "cuda":
        results["infra"]["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6

    # Save results
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


def run_sweep(configs: list, dry_run: bool = False, device: str = "auto"):
    """Run multiple experiment configs, skipping already-completed ones."""
    if not configs:
        logger.info("No configs to run.")
        return

    results_dir = configs[0].logging.results_dir
    completed = find_completed_runs(results_dir)

    pending = []
    for cfg in configs:
        h = cfg.config_hash()
        if h in completed:
            logger.info(f"Skipping (already done): {cfg.name} [{h}]")
        else:
            pending.append(cfg)

    logger.info(f"Sweep: {len(configs)} total, {len(configs) - len(pending)} done, {len(pending)} pending")

    if dry_run:
        print("\n--- DRY RUN: would execute ---")
        for cfg in pending:
            print(f"  {cfg.name} [{cfg.config_hash()}]")
            print(f"    sae: {cfg.sae}")
            print(f"    train: {cfg.train}")
            print(f"    infra: {cfg.infra.gpu_type}")
            print()
        return

    results = []
    for i, cfg in enumerate(pending):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i+1}/{len(pending)}: {cfg.name}")
        logger.info(f"{'='*60}")
        try:
            r = run_single_experiment(cfg, device=device)
            results.append(r)
        except Exception as e:
            logger.error(f"Failed: {cfg.name} — {e}")
            results.append({"name": cfg.name, "error": str(e)})

    # Print summary table
    print_summary(results)


def print_summary(results: list):
    """Print a summary table of completed experiments."""
    if not results:
        return

    print(f"\n{'='*100}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    header = f"{'Name':<40} {'Type':<8} {'Exp.F':>5} {'K':>4} {'Params':>10} {'Time(s)':>8} {'Samp/s':>8} {'Cost($)':>8}"
    print(header)
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r.get('name', '?'):<40} ERROR: {r['error']}")
            continue
        m = r.get("model", {})
        t = r.get("timing", {})
        print(
            f"{r.get('config', {}).get('name', '?'):<40} "
            f"{m.get('sae_type', '?'):<8} "
            f"{m.get('expansion_factor', '?'):>5} "
            f"{m.get('k', '?'):>4} "
            f"{m.get('n_params', 0):>10_} "
            f"{t.get('total_training_time_s', 0):>8.1f} "
            f"{t.get('avg_throughput_samples_sec', 0):>8.0f} "
            f"{t.get('cost_estimate_usd', 0):>8.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Run SAE experiments from YAML configs")
    parser.add_argument("config", help="Path to YAML experiment config")
    parser.add_argument("--sweep", help="Path to YAML sweep config (cartesian product)")
    parser.add_argument("--override", action="append", default=[], help="Override config values (e.g. sae.k=128)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for ov in args.override:
        if "=" not in ov:
            parser.error(f"Override must be key=value, got: {ov}")
        k, v = ov.split("=", 1)
        overrides[k] = v

    # Load base config
    config = load_config(args.config, overrides if overrides else None)

    if args.sweep:
        # Load sweep params and generate configs
        with open(args.sweep) as f:
            sweep_data = yaml.safe_load(f)
        sweep_params = sweep_data.get("sweep", {})
        configs = generate_sweep_configs(config, sweep_params)
        logger.info(f"Generated {len(configs)} sweep configs from {args.sweep}")
        run_sweep(configs, dry_run=args.dry_run, device=args.device)
    else:
        if args.dry_run:
            print(f"Would run: {config.name} [{config.config_hash()}]")
            print(f"  sae: {config.sae}")
            print(f"  train: {config.train}")
            return
        result = run_single_experiment(config, device=args.device)
        print_summary([result])


if __name__ == "__main__":
    main()
