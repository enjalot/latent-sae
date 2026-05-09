"""Disk-backed pooled-embedding SAE trainer.

Adapts `train_otf.py` to read pre-computed pooled embeddings (e.g.
jina-v5-nano fp16 .npy memmaps) instead of producing per-token vectors
on-the-fly. Same training loop, same config knobs, same artifacts —
just a different data source.

Usage:
    python train_pooled.py --config experiments/configs/jina_phase1_24K_oldrecipe_replay4.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from latentsae.sae import Sae
from latentsae.utils.config import SaeConfig, SaeType
from latentsae.utils import data_loader as _dl
from latentsae.utils.data_loader import ShardedEmbeddingDataset
from latentsae.utils.eleuther import geometric_median

# Force every shard to be served via lazy np.memmap (never materialized
# as an in-RAM fp32 tensor). Our jina-v5-nano fp16 shards are 215-242 MB
# — just under the library's 256 MiB lazy threshold — so without this
# override they would each be cast fp16->fp32 (~470 MB/shard) into a
# per-process LRU. With num_workers > 0 each forked worker builds its
# own LRU; that combination OOM'd at 31 GB anon-rss per worker on
# 2026-05-03 10:20 UTC. Lazy mmap costs ~one mmap page-fault per row
# but pages are shared across workers via fork + page cache.
_dl._LARGE_SHARD_BYTES = 0


def train_pooled(cfg: dict):
    out_root = Path(cfg["logging"]["results_dir"])
    run_dir = out_root / f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    device = torch.device(cfg.get("device", "cuda"))
    sae_cfg_dict = cfg["sae"]
    sae_type = SaeType(sae_cfg_dict.get("sae_type", "topk"))
    scfg = SaeConfig(
        sae_type=sae_type,
        expansion_factor=sae_cfg_dict["expansion_factor"],
        k=sae_cfg_dict["k"],
        k_anneal=sae_cfg_dict.get("k_anneal", False),
        matryoshka_sizes=sae_cfg_dict.get("matryoshka_sizes", ""),
        matryoshka_ks=sae_cfg_dict.get("matryoshka_ks", ""),
        matryoshka_use_batchtopk=sae_cfg_dict.get("matryoshka_use_batchtopk", False),
        decoder_init_norm=float(sae_cfg_dict.get("decoder_init_norm", 0.0)),
        normalize_decoder=sae_cfg_dict.get("normalize_decoder", True),
    )
    d_in = cfg["data"]["d_in"]
    sae = Sae(d_in=d_in, cfg=scfg, device=device, dtype=torch.float32)
    print(f"SAE: {sae_type.value}, d_in={d_in}, "
          f"num_latents={sae.num_latents}, params={sum(p.numel() for p in sae.parameters()):,}")

    # Build dataset over one or many embedding shard directories
    data_dirs = cfg["data"]["data_dirs"]
    dataset = ShardedEmbeddingDataset(
        data_dirs,
        cache_size=int(cfg["data"].get("cache_size", 4)),
        d_in=d_in,
        shuffle=False,                 # batch-level random sampler shuffles
        warm_up_cache=False,
        file_type=cfg["data"].get("file_type", "npy"),
    )

    batch_size = cfg["train"]["batch_size"]
    replay_factor = float(cfg["data"].get("replay_factor", 1.0))
    n_samples = int(len(dataset) * replay_factor)
    n_batches = n_samples // batch_size
    print(f"data: {len(dataset):,} unique × {replay_factor} replay = {n_samples:,} samples "
          f"-> {n_batches:,} batches at bs={batch_size}")

    sampler = RandomSampler(dataset, replacement=True, num_samples=n_samples,
                            generator=torch.Generator().manual_seed(cfg["data"].get("seed", 42)))
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(cfg["train"].get("num_workers", 4)) > 0,
    )

    # Optimizer + AMP (match train_otf: Adam (no weight decay), same LR formula)
    lr = cfg["train"].get("lr") or 2e-4 / (sae.num_latents / (2 ** 14)) ** 0.5
    print(f"lr={lr:.2e}  warmup_steps={cfg['train'].get('lr_warmup_steps', 1000)}")
    try:
        from bitsandbytes.optim import Adam8bit as _Adam
        print("Using 8-bit Adam from bitsandbytes")
    except Exception:
        from torch.optim import Adam as _Adam
        print("Using torch.optim.Adam")
    opt = _Adam(sae.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"].get("use_amp", True))

    warmup_steps = cfg["train"].get("lr_warmup_steps", 1000)
    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, n_batches - warmup_steps)
        return lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

    save_every = cfg["train"].get("save_every", 10_000)
    auxk_alpha = cfg["train"].get("auxk_alpha", 0.0)
    dead_feature_threshold = cfg["train"].get("dead_feature_threshold", 50_000)
    use_b_dec_init = cfg["train"].get("use_b_dec_init", True)
    use_decoder_norm = cfg["train"].get("use_decoder_norm", True)
    use_grad_clip = cfg["train"].get("use_grad_clip", True)
    use_remove_parallel_grad = cfg["train"].get("use_remove_parallel_grad", True)
    print(f"fixes: b_dec_init={use_b_dec_init} decoder_norm={use_decoder_norm} "
          f"grad_clip={use_grad_clip} remove_parallel={use_remove_parallel_grad}")
    bias_initialized = not use_b_dec_init
    num_tokens_since_fired = torch.zeros(sae.num_latents, device=device, dtype=torch.long)

    t0 = time.monotonic()
    pbar = tqdm(enumerate(dl), total=n_batches, desc="pooled SAE")
    samples_seen = 0
    for step, batch in pbar:
        if step >= n_batches:
            break
        batch = batch.to(device, non_blocking=True)
        samples_seen += int(batch.shape[0])

        if not bias_initialized:
            with torch.no_grad():
                median = geometric_median(batch)
                sae.b_dec.data = median.to(sae.dtype)
            bias_initialized = True

        if use_decoder_norm and scfg.normalize_decoder:
            sae.set_decoder_norm_to_unit_norm()

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["train"].get("use_amp", True)):
            dead_mask = (num_tokens_since_fired > dead_feature_threshold) if auxk_alpha > 0 else None
            out = sae(batch, dead_mask=dead_mask)
            loss = out.fvu + auxk_alpha * out.auxk_loss
            if scfg.multi_topk:
                loss = loss + out.multi_topk_fvu / 8
        scaler.scale(loss).backward()
        if use_grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            if use_remove_parallel_grad and scfg.normalize_decoder:
                sae.remove_gradient_parallel_to_decoder_directions()
        scaler.step(opt)
        scaler.update()

        if auxk_alpha > 0:
            with torch.no_grad():
                fired = torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
                fired[out.latent_indices.flatten()] = True
                num_tokens_since_fired += int(batch.shape[0])
                num_tokens_since_fired[fired] = 0
        if step % 100 == 0:
            elapsed = max(time.monotonic() - t0, 1e-6)
            pbar.set_postfix({
                "fvu": f"{out.fvu.item():.4f}",
                "samp/s": f"{samples_seen/elapsed:.0f}",
            })
        if step > 0 and (step % save_every == 0):
            sae.save_to_disk(ckpt_dir / f"sae_step_{step}")

    wall = time.monotonic() - t0

    ckpt_name = f"sae_{sae_type.value}_{scfg.k}_{scfg.expansion_factor}.pooled"
    sae.save_to_disk(ckpt_dir / ckpt_name)

    results = {
        "config": cfg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {"n_samples": samples_seen, "d_in": d_in,
                 "source": "pooled_disk",
                 "data_dirs": data_dirs,
                 "n_unique": len(dataset),
                 "replay_factor": replay_factor},
        "model": {
            "sae_type": sae_type.value,
            "expansion_factor": scfg.expansion_factor,
            "k": scfg.k,
            "n_params": sum(p.numel() for p in sae.parameters()),
        },
        "timing": {
            "total_training_time_s": wall,
            "avg_throughput_samples_sec": samples_seen / max(wall, 1e-6),
        },
        "infra": {"gpu_type": torch.cuda.get_device_name(0), "device": "cuda"},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {run_dir}")
    print(f"final throughput: {samples_seen/max(wall,1e-6):,.0f} samples/s "
          f"({samples_seen:,} samples in {wall/60:.1f} min)")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_pooled(cfg)


if __name__ == "__main__":
    main()
