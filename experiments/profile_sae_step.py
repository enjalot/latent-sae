"""Profile the SAE training step at various batch sizes.

Two modes:
  --mode sae_only  : synthetic GPU input, times SAE forward+backward+opt.
                     Isolates compute-only cost.
  --mode full      : spins up the OTF producer and consumes from it.
                     Matches real training wall time.

Outputs:
  - per-step timing (min / p50 / p90 / mean) via torch.cuda.Event
  - breakdown by op-category from torch.profiler (CUDA self-time, top N)
  - estimated samples/sec achievable at each batch size

Keeps the rest of the system untouched (no writes, no Ollama).
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, '/home/enjalot/code/latent-sae')
from latentsae.sae import Sae
from latentsae.utils.config import SaeConfig, SaeType
from latentsae.utils.eleuther import geometric_median
from latentsae.utils.otf_dataset import OnTheFlyColBERTDataset


def build_sae(sizes: str, ks: str, expansion: int, device: str) -> Sae:
    scfg = SaeConfig(
        sae_type=SaeType('matryoshka'),
        expansion_factor=expansion,
        k=32,
        matryoshka_sizes=sizes,
        matryoshka_ks=ks,
    )
    return Sae(d_in=64, cfg=scfg, device=device, dtype=torch.float32)


def train_one_step(sae, opt, scaler, batch, clip=True, dnorm=True, parallel=True):
    """One full step matching train_otf.py (all 8 fixes active)."""
    if dnorm:
        sae.set_decoder_norm_to_unit_norm()
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=True):
        out = sae(batch, dead_mask=None)
        loss = out.fvu
    scaler.scale(loss).backward()
    if clip:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        if parallel:
            sae.remove_gradient_parallel_to_decoder_directions()
    scaler.step(opt)
    scaler.update()
    return out.fvu.item()


def time_steps(sae, opt, scaler, get_batch_fn, n_warmup: int, n_measure: int):
    """Returns list of per-step wall times in ms using cuda events."""
    # Warmup
    for _ in range(n_warmup):
        train_one_step(sae, opt, scaler, get_batch_fn())
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    for i in range(n_measure):
        starts[i].record()
        train_one_step(sae, opt, scaler, get_batch_fn())
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def torch_profile(sae, opt, scaler, get_batch_fn, n_steps: int = 20):
    """Run torch.profiler for n_steps and return the top ops by self CUDA time."""
    from torch.profiler import profile, ProfilerActivity
    # Warmup (outside profiler)
    for _ in range(5):
        train_one_step(sae, opt, scaler, get_batch_fn())
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(n_steps):
            train_one_step(sae, opt, scaler, get_batch_fn())
        torch.cuda.synchronize()
    # Aggregate
    events = prof.key_averages()
    by_cat = defaultdict(lambda: 0.0)
    # Pick attr (torch >= 2.4 renamed self_cuda_time_total → self_device_time_total)
    def _cuda_us(e):
        for a in ('self_device_time_total', 'self_cuda_time_total'):
            if hasattr(e, a):
                return getattr(e, a)
        return 0
    for e in events:
        name = e.key
        if 'matmul' in name or 'gemm' in name or 'mm' in name.lower():
            cat = 'matmul'
        elif 'topk' in name.lower():
            cat = 'topk'
        elif 'triton' in name.lower() or 'sparse_dense' in name.lower():
            cat = 'triton_decode'
        elif 'adam' in name.lower():
            cat = 'optimizer'
        elif 'copy' in name.lower() or 'memcpy' in name.lower():
            cat = 'memcpy'
        elif 'elementwise' in name.lower() or 'unary' in name.lower() or 'binary' in name.lower():
            cat = 'elementwise'
        elif 'norm' in name.lower():
            cat = 'norm_clip'
        else:
            cat = 'other'
        by_cat[cat] += _cuda_us(e) / 1000  # us → ms
    total = sum(by_cat.values())
    return {cat: (ms, 100 * ms / total if total else 0) for cat, ms in sorted(by_cat.items(), key=lambda kv: -kv[1])}, total


def run_sae_only(args):
    torch.set_float32_matmul_precision('high')
    sae = build_sae(args.sizes, args.ks, args.expansion, 'cuda')
    opt = torch.optim.Adam(sae.parameters(), lr=2.83e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # Prebuild fake batch pool (already-on-device) per batch size
    results = []
    for bs in args.batch_sizes:
        # Use a ring of a few batches to avoid caching the same tensor
        pool = [torch.randn(bs, 64, device='cuda', dtype=torch.float32) for _ in range(4)]
        counter = [0]
        def get_batch():
            b = pool[counter[0] % len(pool)]
            counter[0] += 1
            return b
        times_ms = time_steps(sae, opt, scaler, get_batch, args.warmup, args.measure)
        prof_cats, prof_total = torch_profile(sae, opt, scaler, get_batch, n_steps=20)
        results.append({
            'batch_size': bs,
            'times_ms': times_ms,
            'median_ms': float(np.median(times_ms)),
            'mean_ms': float(np.mean(times_ms)),
            'p90_ms': float(np.percentile(times_ms, 90)),
            'min_ms': float(np.min(times_ms)),
            'samples_per_s': bs / (float(np.median(times_ms)) / 1000),
            'profile_cats': prof_cats,
            'profile_total_ms': prof_total / 20,  # per-step
        })

    _print_results(results, title='=== SAE-only (synthetic GPU input) ===')
    return results


def run_full(args):
    torch.set_float32_matmul_precision('high')
    sae = build_sae(args.sizes, args.ks, args.expansion, 'cuda')
    opt = torch.optim.Adam(sae.parameters(), lr=2.83e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    results = []
    for bs in args.batch_sizes:
        ds = OnTheFlyColBERTDataset(
            parquet_dirs=[
                '/data/chunks/fineweb-edu-sample-10BT-chunked-500/train',
                '/data/chunks/RedPajama-Data-V2-sample-10B-chunked-500/train',
                '/data/chunks/pile-uncopyrighted-chunked-500/train',
            ],
            batch_size=bs, chunks_per_encode=64, queue_size=4, device='cuda', seed=42,
            shuffle_buffer_size=4_000_000,
            on_device=args.on_device)
        it = iter(ds)
        if args.on_device:
            def get_batch(): return next(it)
        else:
            def get_batch(): return next(it).to('cuda', non_blocking=True)
        # Longer warmup for full because ColBERT + shuffle buffer need time to reach steady state
        times_ms = time_steps(sae, opt, scaler, get_batch, args.warmup + 20, args.measure)
        prof_cats, prof_total = torch_profile(sae, opt, scaler, get_batch, n_steps=20)
        results.append({
            'batch_size': bs,
            'times_ms': times_ms,
            'median_ms': float(np.median(times_ms)),
            'mean_ms': float(np.mean(times_ms)),
            'p90_ms': float(np.percentile(times_ms, 90)),
            'min_ms': float(np.min(times_ms)),
            'samples_per_s': bs / (float(np.median(times_ms)) / 1000),
            'profile_cats': prof_cats,
            'profile_total_ms': prof_total / 20,
        })
        ds.stop()

    _print_results(results, title='=== Full pipeline (OTF + SAE) ===')
    return results


def _print_results(results, title):
    print()
    print(title)
    print(f"{'bs':>6}  {'med_ms':>7}  {'p90_ms':>7}  {'min_ms':>7}  {'samples/s':>11}  {'rel':>5}")
    base = results[0]['samples_per_s']
    for r in results:
        rel = r['samples_per_s'] / base
        print(f"{r['batch_size']:>6}  {r['median_ms']:>7.2f}  {r['p90_ms']:>7.2f}  {r['min_ms']:>7.2f}  {r['samples_per_s']:>11,.0f}  {rel:>4.2f}x")
    print()
    print('=== kernel breakdown (per step, in ms, torch.profiler) ===')
    print(f"{'bs':>6}  {'total':>7}  categories (ms, %)")
    for r in results:
        s = f"{r['batch_size']:>6}  {r['profile_total_ms']:>7.2f}  "
        parts = [f"{cat}: {ms:.2f}ms ({pct:.0f}%)" for cat, (ms, pct) in list(r['profile_cats'].items())[:6]]
        print(s + '  |  '.join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sae_only', 'full', 'both'], default='both')
    ap.add_argument('--sizes', default='512,2048,8192,16384,32768',
                    help='matryoshka sizes (match Phase 17 arch)')
    ap.add_argument('--ks', default='8,16,32,32,32')
    ap.add_argument('--expansion', type=int, default=512)
    ap.add_argument('--batch-sizes', type=int, nargs='+',
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument('--warmup', type=int, default=15)
    ap.add_argument('--measure', type=int, default=50)
    ap.add_argument('--on-device', action='store_true',
                    help='use the new OTF on-device (GPU-tensor) pipeline')
    args = ap.parse_args()

    if args.mode in ('sae_only', 'both'):
        run_sae_only(args)
        if args.mode == 'both':
            # Free before spinning up ColBERT
            torch.cuda.empty_cache()
    if args.mode in ('full', 'both'):
        run_full(args)


if __name__ == '__main__':
    main()
