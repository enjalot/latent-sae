"""Domain probe v2 — rank features by DISTINCTIVENESS per domain.

The v1 domain_probe showed the top features by raw activation sum were
identical across all domains (a few "always on" features dominate). This
version ranks features by z-score relative to a corpus-wide baseline,
surfacing features that are specifically more active in a given domain.

Usage: same CLI as domain_probe.py.
"""
import argparse
import json
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402

from experiments.domain_probe import (DATASET_PATHS, DOMAINS,
                                       find_domain_chunks, load_chunk_texts)


@torch.no_grad()
def feature_stats_for_chunks(sae: Sae, vectors: np.ndarray, offsets: np.ndarray,
                             chunk_idxs: list[int], device: str = "cuda"):
    """Return (mean_act_per_feature, fraction_of_tokens_active_per_feature)."""
    sae.eval()
    N = sae.num_latents
    act_sum = torch.zeros(N, device=device)
    act_count = torch.zeros(N, device=device, dtype=torch.int64)
    total_tokens = 0
    for ci in chunk_idxs:
        s, e = int(offsets[ci]), int(offsets[ci + 1])
        if e <= s:
            continue
        total_tokens += e - s
        batch = torch.from_numpy(
            np.ascontiguousarray(vectors[s:e], dtype=np.float32)
        ).to(device)
        out = sae(batch)
        acts = out.latent_acts
        idxs = out.latent_indices
        mask = acts > 0
        idxs_flat = idxs[mask].long()
        acts_flat = acts[mask]
        act_sum.scatter_add_(0, idxs_flat, acts_flat)
        act_count.scatter_add_(0, idxs_flat, torch.ones_like(idxs_flat, dtype=torch.int64))
    mean_act = (act_sum / max(total_tokens, 1)).cpu().numpy()
    act_rate = (act_count.float() / max(total_tokens, 1)).cpu().numpy()
    return mean_act, act_rate, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", action="append", default=[])
    ap.add_argument("--dataset", default="fineweb")
    ap.add_argument("--n-chunks-per-domain", type=int, default=40)
    ap.add_argument("--top-k-features", type=int, default=12)
    ap.add_argument("--n-candidate-chunks", type=int, default=20000)
    ap.add_argument("--n-baseline-chunks", type=int, default=2000,
                    help="random chunks from the full candidate pool used as baseline")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",
                    default="/data/embeddings/beir/domain_probe_distinctive.json")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for pat in args.sae_dir:
        expanded = sorted(Path(p) for p in glob(pat))
        if expanded:
            run_dirs.extend(expanded)
        else:
            p = Path(pat)
            if p.exists():
                run_dirs.append(p)

    paths = DATASET_PATHS[args.dataset]
    print(f"loading up to {args.n_candidate_chunks:,} chunks...")
    chunk_texts = load_chunk_texts(paths["parquets"], args.n_candidate_chunks)
    offsets = np.load(Path(paths["vectors"]) / "chunk_offsets.npy")
    vec_file = next(Path(paths["vectors"]).glob("data-*.npy"))
    vectors = np.load(vec_file, mmap_mode="r")

    # Find chunks per domain and baseline
    print("finding per-domain chunks...")
    domain_chunks: dict[str, list[int]] = {}
    for name, seed in DOMAINS.items():
        idxs = find_domain_chunks(seed, chunk_texts, n=args.n_chunks_per_domain)
        idxs = [i for i in idxs if i < len(offsets) - 1]
        domain_chunks[name] = idxs

    rng = np.random.default_rng(42)
    baseline_idxs = rng.choice(min(args.n_candidate_chunks,
                                    len(offsets) - 1),
                                size=args.n_baseline_chunks, replace=False).tolist()
    # Exclude any that landed in a domain (very unlikely overlap but be safe)
    domain_pool = set(ci for v in domain_chunks.values() for ci in v)
    baseline_idxs = [i for i in baseline_idxs if i not in domain_pool]
    print(f"  baseline: {len(baseline_idxs):,} random chunks")

    results = {"domains": domain_chunks, "baseline_size": len(baseline_idxs),
               "saes": {}}
    for rd in run_dirs:
        ckpt = next((p for p in (rd / "checkpoints").glob("*")
                     if p.is_dir() and (p / "cfg.json").exists()), None)
        if ckpt is None:
            print(f"SKIP {rd.name}: no checkpoint")
            continue
        print(f"\n== {rd.name} ==")
        sae = Sae.load_from_disk(ckpt, device=args.device)

        # Baseline stats
        t = time.monotonic()
        base_mean, base_rate, base_tokens = feature_stats_for_chunks(
            sae, vectors, offsets, baseline_idxs, device=args.device)
        # Tiny epsilon so z-score is stable
        eps = 1e-6

        per_sae: dict[str, list[dict]] = {}
        for name, idxs in domain_chunks.items():
            dom_mean, dom_rate, _ = feature_stats_for_chunks(
                sae, vectors, offsets, idxs, device=args.device)
            # Score = distinctiveness: mean activation lift + rate lift, normalized
            mean_lift = (dom_mean - base_mean) / (base_mean + eps)
            rate_lift = (dom_rate - base_rate) / (base_rate + eps)
            # Combined z-like score; require feature fires meaningfully in domain
            mask = dom_rate > 0.001
            score = np.where(mask, mean_lift + rate_lift, -np.inf)
            order = np.argsort(-score)[: args.top_k_features]
            per_sae[name] = [
                {"feature": int(fid),
                 "domain_mean_act": float(dom_mean[fid]),
                 "baseline_mean_act": float(base_mean[fid]),
                 "domain_rate": float(dom_rate[fid]),
                 "baseline_rate": float(base_rate[fid]),
                 "score": float(score[fid])}
                for fid in order if np.isfinite(score[fid])
            ]
            # Preview: just list the feature ids
            ids = [h["feature"] for h in per_sae[name]]
            print(f"  {name:<24} distinctive top-{args.top_k_features}: {ids}  ({time.monotonic() - t:.1f}s)")
        results["saes"][rd.name] = {
            "checkpoint": ckpt.name,
            "num_latents": sae.num_latents,
            "sae_type": sae.cfg.sae_type.value,
            "per_domain": per_sae,
        }
        del sae
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
