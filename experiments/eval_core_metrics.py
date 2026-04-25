"""Post-hoc 'core metrics' eval for trained SAEs.

The trainer only logs FVU/L0/dead to wandb. When wandb is disabled we need
these numbers after the fact. This script walks a directory of run-dirs,
loads each checkpoint, forwards a bounded held-out sample of embedding
vectors through the SAE, and writes a metrics.json alongside the run's
results.json.

Metrics computed:
    fvu               — 1 - Var(err) / Var(x), averaged element-wise
    mse               — mean squared error per element
    l0                — expected number of active latents per sample (== k for TopK)
    dead_features     — # latents that never fired on the eval sample
    dead_pct          — dead_features / num_latents
    mmcs              — mean max cosine similarity of decoder rows (feature uniqueness;
                         lower = more decorrelated features)

Usage:
    python -m experiments.eval_core_metrics experiments/results \\
        --eval-data /data/embeddings/.../train \\
        --eval-offset 100_000_000 --n-eval 200_000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


def load_eval_sample(data_dir: str, offset: int, n: int, d_in: int) -> torch.Tensor:
    """Read `n` rows starting at `offset` from the first .npy shard in data_dir."""
    d = Path(data_dir)
    shards = sorted(p for p in d.glob("data-*.npy"))
    if not shards:
        raise FileNotFoundError(f"no data-*.npy in {d}")
    arr = np.load(shards[0], mmap_mode="r")
    total = arr.shape[0]
    assert arr.shape[1] == d_in, f"d_in mismatch: {arr.shape[1]} vs {d_in}"
    end = min(offset + n, total)
    start = max(offset, 0)
    if end <= start:
        raise ValueError(f"empty eval slice: offset={offset}, n={n}, total={total}")
    chunk = np.asarray(arr[start:end], dtype=np.float32)
    return torch.from_numpy(chunk)


def mmcs(W: torch.Tensor, sample_size: int = 4096) -> float:
    """Mean max cosine similarity between decoder rows, excluding self.

    For large latent counts we subsample rows to keep this under a second.
    """
    W = torch.nn.functional.normalize(W, dim=-1)
    n = W.shape[0]
    if n > sample_size:
        idx = torch.randperm(n, device=W.device)[:sample_size]
        W = W[idx]
    sim = W @ W.T
    sim.fill_diagonal_(-1.0)
    return sim.max(dim=-1).values.mean().item()


@torch.no_grad()
def eval_checkpoint(ckpt_dir: Path, x: torch.Tensor, batch_size: int = 4096,
                    device: str = "cuda") -> dict:
    sae = Sae.load_from_disk(ckpt_dir, device=device)
    sae.eval()
    num_latents = sae.num_latents

    n = x.shape[0]
    fire_count = torch.zeros(num_latents, device=device, dtype=torch.long)
    l0_sum = 0.0
    sq_err_sum = 0.0
    x_sq_sum = 0.0
    x_sum = 0.0
    n_elem = 0

    for s in range(0, n, batch_size):
        batch = x[s:s + batch_size].to(device, dtype=sae.dtype)
        out = sae(batch)
        recon = out.sae_out
        latent_indices = out.latent_indices   # (B, k)
        latent_acts = out.latent_acts         # (B, k)

        err = (batch - recon).float()
        sq_err_sum += (err ** 2).sum().item()
        x_sum += batch.float().sum().item()
        x_sq_sum += (batch.float() ** 2).sum().item()
        n_elem += batch.numel()

        active = latent_acts > 0              # (B, k)
        l0_sum += active.sum().item()
        fired_ids = latent_indices[active].long()
        fire_count.scatter_add_(
            0, fired_ids,
            torch.ones_like(fired_ids, dtype=torch.long))

    mse = sq_err_sum / n_elem
    x_var = (x_sq_sum / n_elem) - (x_sum / n_elem) ** 2
    fvu = mse / max(x_var, 1e-12)
    l0 = l0_sum / n
    dead = (fire_count == 0).sum().item()
    dead_pct = dead / num_latents

    return {
        "fvu": fvu,
        "mse": mse,
        "l0": l0,
        "dead_features": int(dead),
        "dead_pct": dead_pct,
        "num_latents": int(num_latents),
        "mmcs": mmcs(sae.W_dec.float()) if sae.cfg.sae_type.value != "gated" else mmcs(sae.W_dec.float()),
        "n_eval": int(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="experiments/results")
    ap.add_argument("--eval-data", required=True,
                    help="path to embedding dir (must contain data-*.npy)")
    ap.add_argument("--eval-offset", type=int, default=100_000_000,
                    help="row offset — defaults to just past the 100M training slice")
    ap.add_argument("--n-eval", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--pattern", default="colbert_mxbai_phase1*",
                    help="glob pattern under results_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"loading {args.n_eval:,} eval vectors from {args.eval_data} "
          f"offset={args.eval_offset:,}")
    x = load_eval_sample(args.eval_data, args.eval_offset, args.n_eval, args.d_in)
    print(f"  shape: {tuple(x.shape)}, dtype: {x.dtype}")

    run_dirs = sorted(Path(args.results_dir).glob(args.pattern))
    print(f"found {len(run_dirs)} run(s) matching {args.pattern!r}\n")

    rows = []
    for rd in run_dirs:
        # Find single checkpoint subdirectory. Accept with-or-without trailing dot.
        ckpt_dirs = [p for p in (rd / "checkpoints").glob("*") if p.is_dir() and (p / "cfg.json").exists()]
        if not ckpt_dirs:
            print(f"SKIP {rd.name}: no checkpoint with cfg.json")
            continue
        ckpt = ckpt_dirs[0]
        out_path = rd / "metrics.json"
        if out_path.exists() and not args.overwrite:
            m = json.loads(out_path.read_text())
            print(f"cached {rd.name}: {m}")
            rows.append((rd.name, m))
            continue

        print(f"eval  {rd.name} / {ckpt.name}")
        m = eval_checkpoint(ckpt, x, batch_size=args.batch_size, device=args.device)
        out_path.write_text(json.dumps(m, indent=2))
        print(f"  {m}")
        rows.append((rd.name, m))

    # Pretty summary — pull k/exp/auxk from run dir name when present
    def _parse_label(name: str) -> tuple:
        def _grab(tag: str, default):
            if tag not in name:
                return default
            chunk = name.split(tag, 1)[1].split("_")[0]
            try:
                return type(default)(chunk)
            except ValueError:
                return default
        k = _grab("_k", -1)
        ef = _grab("expansion_factor", -1)
        auxk = _grab("auxk_alpha", -1.0)
        return k, ef, auxk

    print("\n=== SUMMARY ===")
    print(f"{'run':70}  {'k':>4}  {'exp':>4}  {'auxk':>7}  {'FVU':>9}  {'L0':>6}  "
          f"{'dead%':>6}  {'MMCS':>6}")
    for name, m in rows:
        k, ef, auxk = _parse_label(name)
        print(f"{name:70}  {k:>4}  {ef:>4}  {auxk:>7.4g}  {m['fvu']:>9.5f}  "
              f"{m['l0']:>6.2f}  {m['dead_pct']*100:>5.1f}%  {m['mmcs']:>6.3f}")


if __name__ == "__main__":
    main()
