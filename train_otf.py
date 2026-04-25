"""On-the-fly ColBERT → SAE trainer.

Bypasses the `SaeTrainer`/`DataLoader` machinery to consume batches
directly from `OnTheFlyColBERTDataset`. Produces the same output
artifacts (`results.json`, `metrics.json` stub, checkpoint) so the
existing extraction / labeling / retrieval eval pipeline works with
no changes.

Usage:
    python train_otf.py --config experiments/configs/colbert_phase16_otf.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from latentsae.sae import Sae
from latentsae.utils.config import SaeConfig, SaeType
from latentsae.utils.eleuther import geometric_median
from latentsae.utils.otf_dataset import OnTheFlyColBERTDataset


def train_otf(cfg: dict):
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

    # Optimizer + AMP (match SaeTrainer: Adam (no weight decay), same LR formula)
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

    batch_size = cfg["train"]["batch_size"]
    n_samples = cfg["data"]["n_samples"]
    n_batches = (n_samples + batch_size - 1) // batch_size

    dataset = OnTheFlyColBERTDataset(
        parquet_dirs=cfg["data"]["parquet_dirs"],
        model_id=cfg["data"].get("model_id",
                                 "mixedbread-ai/mxbai-edge-colbert-v0-32m"),
        batch_size=batch_size,
        chunks_per_encode=cfg["data"].get("chunks_per_encode", 64),
        device=device,
        queue_size=cfg["data"].get("queue_size", 4),
        seed=cfg["data"].get("seed", 42),
        domain_weights=cfg["data"].get("domain_weights"),
        shuffle_buffer_size=int(cfg["data"].get("shuffle_buffer_size", 0)),
        replay_factor=float(cfg["data"].get("replay_factor", 1.0)),
        on_device=bool(cfg["data"].get("on_device", False)),
    )

    # Warmup schedule
    warmup_steps = cfg["train"].get("lr_warmup_steps", 1000)
    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        # Cosine to 10% lr over remaining steps
        progress = (step - warmup_steps) / max(1, n_batches - warmup_steps)
        return lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

    t0 = time.monotonic()
    logs = []
    pbar = tqdm(enumerate(dataset), total=n_batches, desc="OTF SAE")
    save_every = cfg["train"].get("save_every", 10_000)
    auxk_alpha = cfg["train"].get("auxk_alpha", 0.0)
    dead_feature_threshold = cfg["train"].get("dead_feature_threshold", 50_000)
    # Feature flags for A/B testing trainer niceties (default: all on)
    use_b_dec_init = cfg["train"].get("use_b_dec_init", True)
    use_decoder_norm = cfg["train"].get("use_decoder_norm", True)
    use_grad_clip = cfg["train"].get("use_grad_clip", True)
    use_remove_parallel_grad = cfg["train"].get("use_remove_parallel_grad", True)
    print(f"fixes: b_dec_init={use_b_dec_init} decoder_norm={use_decoder_norm} "
          f"grad_clip={use_grad_clip} remove_parallel={use_remove_parallel_grad}")
    bias_initialized = not use_b_dec_init
    num_tokens_since_fired = torch.zeros(sae.num_latents, device=device, dtype=torch.long)
    for step, batch in pbar:
        if step >= n_batches:
            break
        batch = batch.to(device, non_blocking=True)

        # Initialize decoder bias to geometric median of first batch
        # (matches SaeTrainer in trainer.py — centers the SAE input
        # distribution on the data centroid instead of starting at zero).
        if not bias_initialized:
            with torch.no_grad():
                median = geometric_median(batch)
                sae.b_dec.data = median.to(sae.dtype)
            bias_initialized = True

        # Keep decoder rows at unit norm (matches SaeTrainer).
        if use_decoder_norm and scfg.normalize_decoder:
            sae.set_decoder_norm_to_unit_norm()

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["train"].get("use_amp", True)):
            dead_mask = (num_tokens_since_fired > dead_feature_threshold) if auxk_alpha > 0 else None
            out = sae(batch, dead_mask=dead_mask)
            loss = out.fvu + auxk_alpha * out.auxk_loss
            # Gao et al. matryoshka/multi_topk aux loss (L(k) + L(4k)/8)
            if scfg.multi_topk:
                loss = loss + out.multi_topk_fvu / 8
        scaler.scale(loss).backward()
        # Match SaeTrainer: unscale + grad clip + decoder-parallel grad removal
        if use_grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            if use_remove_parallel_grad and scfg.normalize_decoder:
                sae.remove_gradient_parallel_to_decoder_directions()
        scaler.step(opt)
        scaler.update()

        # Track dead features for auxk revival
        if auxk_alpha > 0:
            with torch.no_grad():
                fired = torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
                fired[out.latent_indices.flatten()] = True
                num_tokens_since_fired += int(batch.shape[0])
                num_tokens_since_fired[fired] = 0
        if step % 100 == 0:
            s = dataset.stats()
            pbar.set_postfix({
                "fvu": f"{out.fvu.item():.4f}",
                "tok/s": f"{s['tokens_per_s']:.0f}",
                "qd": s["queue_depth"],
            })
        if step > 0 and (step % save_every == 0):
            sae.save_to_disk(ckpt_dir / f"sae_step_{step}")

    wall = time.monotonic() - t0
    final_stats = dataset.stats()
    dataset.stop()

    # Final checkpoint matching existing naming convention
    ckpt_name = f"sae_{sae_type.value}_{scfg.k}_{scfg.expansion_factor}.otf"
    sae.save_to_disk(ckpt_dir / ckpt_name)

    # Emit results.json + stub metrics.json so existing pipeline picks it up
    results = {
        "config": cfg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {"n_samples": n_samples, "d_in": d_in,
                 "source": "on_the_fly",
                 "parquet_dirs": cfg["data"]["parquet_dirs"]},
        "model": {
            "sae_type": sae_type.value,
            "expansion_factor": scfg.expansion_factor,
            "k": scfg.k,
            "n_params": sum(p.numel() for p in sae.parameters()),
        },
        "timing": {
            "total_training_time_s": wall,
            "avg_throughput_samples_sec": n_samples / wall,
            "tokens_produced": final_stats["tokens_produced"],
            "otf_tokens_per_sec": final_stats["tokens_per_s"],
        },
        "infra": {"gpu_type": torch.cuda.get_device_name(0), "device": "cuda"},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {run_dir}")
    print(f"final throughput: {n_samples/wall:,.0f} samples/s  (OTF: {final_stats['tokens_per_s']:,.0f} tok/s)")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_otf(cfg)


if __name__ == "__main__":
    main()
