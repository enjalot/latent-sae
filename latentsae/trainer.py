import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .sae import Sae
from .utils.config import TrainConfig, LRSchedule
from .utils.eleuther import geometric_median

GPU_HOURLY_RATES = {"t4": 0.59, "a10g": 1.10, "a100_40gb": 2.10}


class SaeTrainer:
    def __init__(self, cfg: TrainConfig, dataset: Dataset, device: torch.device):
        self.cfg = cfg
        self.dataset = dataset
        self.device = device

        d_in = cfg.d_in
        num_examples = len(dataset)
        self.num_examples = num_examples

        self.sae = Sae(d_in, cfg.sae, device)
        lr = cfg.lr or 2e-4 / (self.sae.num_latents / (2 ** 14)) ** 0.5
        print(f"SAE: {cfg.sae.sae_type.value}, d_in={d_in}, num_latents={self.sae.num_latents}, lr={lr:.2e}")

        try:
            from bitsandbytes.optim import Adam8bit as Adam
            assert torch.cuda.is_available(), "bitsandbytes requires CUDA"
            print("Using 8-bit Adam from bitsandbytes")
        except Exception:
            from torch.optim import Adam
            print("Using torch.optim.Adam (install bitsandbytes for 8-bit Adam)")

        self.optimizer = Adam(self.sae.parameters(), lr=lr)

        total_steps = num_examples // cfg.batch_size
        if cfg.lr_schedule == LRSchedule.COSINE:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup = LinearLR(self.optimizer, start_factor=1e-8, total_iters=cfg.lr_warmup_steps)
            cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - cfg.lr_warmup_steps, 1))
            self.lr_scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[cfg.lr_warmup_steps])
        else:
            from transformers import get_linear_schedule_with_warmup
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, cfg.lr_warmup_steps, total_steps
            )

    def fit(self):
        torch.set_float32_matmul_precision("high")
        cfg = self.cfg

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized()

        wandb = None
        if cfg.log_to_wandb and rank_zero:
            try:
                import wandb as _wandb
                wandb = _wandb
                wandb.init(
                    name=cfg.run_name,
                    project=cfg.wandb_project,
                    config=asdict(cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")

        print(f"SAE parameters: {sum(p.numel() for p in self.sae.parameters()):_}")

        sae = self.sae
        if ddp:
            # Initialize decoder bias before wrapping with DDP
            dl_init = DataLoader(self.dataset, batch_size=cfg.batch_size, num_workers=0)
            first_batch = next(iter(dl_init)).to(self.device)
            median = geometric_median(self._maybe_all_cat(first_batch))
            sae.b_dec.data = median.to(sae.dtype)
            sae = DDP(sae, device_ids=[dist.get_rank()])

        raw = sae.module if isinstance(sae, DDP) else sae

        dl = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        pbar = tqdm(dl, desc="Training", disable=not rank_zero)

        did_fire = torch.zeros(raw.num_latents, device=self.device, dtype=torch.bool)
        num_tokens_since_fired = torch.zeros(raw.num_latents, device=self.device, dtype=torch.long)
        num_tokens_in_step = 0

        avg_auxk_loss = 0.0
        avg_fvu = 0.0
        avg_loss = 0.0
        avg_multi_topk_fvu = 0.0
        last_log_time = time.time()
        training_start = time.time()
        num_params = sum(p.numel() for p in self.sae.parameters())

        use_amp = cfg.use_amp and self.device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)
        bias_initialized = not ddp  # DDP already initialized above

        acc_steps = cfg.grad_acc_steps * cfg.micro_acc_steps
        denom = acc_steps * cfg.wandb_log_frequency
        total_batches = len(dl)

        # -- K-annealing setup --
        # Start with more active features and linearly decrease to target k.
        # More active features early on = richer gradient signal for decoder weights,
        # then tighten sparsity as features stabilize.
        k_anneal = raw.cfg.k_anneal and raw.cfg.sae_type.value != "jumprelu"  # JumpReLU learns its own sparsity
        if k_anneal:
            k_start = raw.cfg.k_anneal_start or (4 * raw.cfg.k)
            k_end = raw.cfg.k
            k_anneal_steps = int(total_batches * raw.cfg.k_anneal_pct)
            print(f"K-annealing: {k_start} -> {k_end} over {k_anneal_steps} batches ({raw.cfg.k_anneal_pct:.0%} of training)")

        for i, batch in enumerate(pbar):
            num_tokens_in_step += batch.numel()
            hiddens = batch.to(self.device)

            # Initialize decoder bias on first batch (non-DDP)
            if not bias_initialized:
                median = geometric_median(self._maybe_all_cat(hiddens))
                raw.b_dec.data = median.to(raw.dtype)
                bias_initialized = True

            if raw.cfg.normalize_decoder:
                raw.set_decoder_norm_to_unit_norm()

            # Compute current k for k-annealing (linear interpolation from k_start to k_end)
            current_k = None
            if k_anneal:
                if i < k_anneal_steps:
                    frac = i / max(k_anneal_steps, 1)
                    current_k = int(k_start + (k_end - k_start) * frac)
                # else: None means use default cfg.k

            for chunk in hiddens.chunk(cfg.micro_acc_steps):
                if torch.isnan(chunk).any():
                    continue

                with autocast(enabled=use_amp):
                    out = sae(
                        chunk,
                        dead_mask=(
                            num_tokens_since_fired > cfg.dead_feature_threshold
                            if cfg.auxk_alpha > 0
                            else None
                        ),
                        k_override=current_k,
                    )

                if torch.isnan(out.fvu):
                    continue

                avg_fvu += float(self._maybe_all_reduce(out.fvu.detach()) / denom)

                if cfg.auxk_alpha > 0:
                    avg_auxk_loss += float(self._maybe_all_reduce(out.auxk_loss.detach()) / denom)

                # -- Reconstruction loss: standard FVU or tilted ERM --
                # Tilted ERM replaces mean loss with (1/t)*log(mean(exp(t*loss_i))),
                # which gently upweights hard-to-reconstruct samples for small t (~2e-3).
                if cfg.tilted_erm_tilt > 0:
                    t = cfg.tilted_erm_tilt
                    per_sample_mse = (out.sae_out - chunk).pow(2).sum(dim=-1)
                    n = torch.tensor(per_sample_mse.shape[0], dtype=per_sample_mse.dtype, device=per_sample_mse.device)
                    recon_loss = (torch.logsumexp(t * per_sample_mse, dim=0) - torch.log(n)) / t
                else:
                    recon_loss = out.fvu

                loss = recon_loss + cfg.auxk_alpha * out.auxk_loss
                if raw.cfg.multi_topk:
                    loss = loss + out.multi_topk_fvu

                avg_loss += float(self._maybe_all_reduce(loss.detach()) / denom)
                if raw.cfg.multi_topk:
                    avg_multi_topk_fvu += float(self._maybe_all_reduce(out.multi_topk_fvu.detach()) / denom)

                scaler.scale(loss / acc_steps).backward()

                did_fire[out.latent_indices.flatten()] = True
                self._maybe_all_reduce(did_fire, "max")

            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

            # Gradient accumulation step
            step, substep = divmod(i + 1, cfg.grad_acc_steps)
            if substep == 0:
                if raw.cfg.normalize_decoder:
                    raw.remove_gradient_parallel_to_decoder_directions()

                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                with torch.no_grad():
                    num_tokens_since_fired += num_tokens_in_step
                    num_tokens_since_fired[did_fire] = 0
                    num_tokens_in_step = 0
                    did_fire.zero_()

                if wandb and (step + 1) % cfg.wandb_log_frequency == 0:
                    now = time.time()
                    elapsed = max(now - last_log_time, 1e-6)
                    samples_logged = cfg.batch_size * cfg.wandb_log_frequency
                    last_log_time = now

                    dead_mask = num_tokens_since_fired > cfg.dead_feature_threshold
                    info = {
                        "fvu": avg_fvu,
                        "loss_total": avg_loss,
                        "dead_pct": dead_mask.float().mean().item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "throughput_samples_sec": samples_logged / elapsed,
                    }
                    if k_anneal and current_k is not None:
                        info["k_current"] = current_k
                    if cfg.auxk_alpha > 0:
                        info["auxk_loss"] = avg_auxk_loss
                    if raw.cfg.multi_topk:
                        info["multi_topk_fvu"] = avg_multi_topk_fvu
                    if self.device.type == "cuda":
                        info["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6

                    avg_auxk_loss = 0.0
                    avg_fvu = 0.0
                    avg_loss = 0.0
                    avg_multi_topk_fvu = 0.0

                    if rank_zero:
                        wandb.log(info, step=step)

                    pbar.set_postfix(fvu=info["fvu"], dead=f"{info['dead_pct']:.1%}")

                if (step + 1) % cfg.save_every == 0:
                    self._save(wandb)

        self._save(wandb)

        # Log final training summary
        total_time = time.time() - training_start
        if wandb and rank_zero:
            summary = {
                "total_training_time_s": total_time,
                "total_samples": self.num_examples,
                "model_params": num_params,
                "avg_throughput_samples_sec": self.num_examples / max(total_time, 1),
            }
            if cfg.gpu_type and cfg.gpu_type in GPU_HOURLY_RATES:
                summary["cost_estimate_usd"] = total_time * GPU_HOURLY_RATES[cfg.gpu_type] / 3600
            wandb.log(summary)
            wandb.summary.update(summary)
        if rank_zero:
            print(f"Training complete: {self.num_examples:_} samples in {total_time:.1f}s "
                  f"({self.num_examples / max(total_time, 1):.0f} samples/sec)")

        pbar.close()

    def _save(self, wandb=None):
        """Save SAE checkpoint to disk."""
        if not dist.is_initialized() or dist.get_rank() == 0:
            sae = self.sae.module if isinstance(self.sae, DDP) else self.sae
            run_id = wandb.run.id if wandb and wandb.run else ""
            name = f"sae_{self.cfg.sae.sae_type.value}_{self.cfg.sae.k}_{self.cfg.sae.expansion_factor}"
            path = self.cfg.checkpoints_directory or "checkpoints"
            sae.save_to_disk(f"{path}/{name}.{run_id}")
            print(f"Saved checkpoint to {path}/{name}.{run_id}")

        if dist.is_initialized():
            dist.barrier()

    def _maybe_all_cat(self, x: Tensor) -> Tensor:
        if not dist.is_initialized():
            return x
        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def _maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized():
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x
