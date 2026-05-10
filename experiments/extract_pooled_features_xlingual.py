"""Cross-lingual top-activating chunks for a pooled-embedding SAE.

Like extract_pooled_features.py, but scans across ALL configured
corpora (3 EN + 20 ML jina-v5-nano) and tags each top-hit with its
source corpus so we can compute per-feature language distributions
afterwards.

Output: feature_activations_xlingual.json with shape:

  {
    "run": "...",
    "checkpoint": "...",
    "num_latents": 24576,
    "n_live_features": ...,
    "live_feature_ids": [...],
    "corpora": {"<corpus_slug>": <n_chunks_scanned>},
    "features": {
      "<fid>": [
        {"activation": float, "corpus": "<slug>", "chunk_idx": int,
         "text": "...", "window": "..."},
        ...
      ]
    }
  }

Usage:
    python -m experiments.extract_pooled_features_xlingual \\
        --sae-dir experiments/results/<run-glob> \\
        --n-per-corpus 20000 --top-n 16
"""
import argparse
import heapq
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


CORPORA = [
    ("fineweb",   "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train",
                  "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
                  "data-*.parquet"),
    ("redpajama", "/data/embeddings/RedPajama-Data-V2-sample-10B-chunked-500-jina-v5-nano/train",
                  "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-500/train",
                  "data-*.parquet"),
    ("pile",      "/data/embeddings/pile-uncopyrighted-chunked-500-jina-v5-nano/train",
                  "/data/chunks/pile-uncopyrighted-chunked-500/train",
                  "data-*.parquet"),
]
ML_LANGS = ["arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
            "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
            "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
            "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn"]
for lang in ML_LANGS:
    slug = f"ml_{lang}"
    CORPORA.append((slug,
                    f"/data/embeddings/fineweb2-{lang}-chunked-500-jina-v5-nano/train",
                    f"/data/chunks/fineweb2-{lang}-chunked-500/train",
                    "*.parquet"))


def choose_checkpoint(run_dir: Path) -> Path:
    ckpt_dirs = [p for p in (run_dir / "checkpoints").glob("*")
                 if p.is_dir() and (p / "cfg.json").exists()]
    if not ckpt_dirs:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")

    def rank(p: Path):
        name = p.name
        if name.startswith("sae_step_"):
            try:
                return (0, int(name.split("_")[-1]))
            except ValueError:
                return (0, 0)
        return (1, 0)

    return sorted(ckpt_dirs, key=rank)[-1]


def list_shards(vec_dir: str) -> list[Path]:
    return sorted(Path(vec_dir).glob("*.npy"))


def load_chunk_texts(parquet_dir: str, glob_pat: str, n_chunks: int) -> list[str]:
    parquets = sorted(Path(parquet_dir).glob(glob_pat))
    if not parquets:
        # multilingual fall-back
        parquets = sorted(Path(parquet_dir).glob("*.parquet"))
    frames, got = [], 0
    for p in parquets:
        df = pd.read_parquet(p, columns=["chunk_text"])
        frames.append(df)
        got += len(df)
        if got >= n_chunks:
            break
    df = pd.concat(frames, ignore_index=True).head(n_chunks)
    return df["chunk_text"].tolist()


@torch.no_grad()
def collect_top_xlingual(sae, corpora, n_per_corpus, top_n,
                         batch_size=8192, device="cuda"):
    """Per-feature min-heap of size top_n over (act, corpus_slug, chunk_idx)."""
    sae.eval()
    num_latents = sae.num_latents
    feat_heaps: list[list[tuple]] = [[] for _ in range(num_latents)]
    counts: dict[str, int] = {}

    t0 = time.monotonic()
    total_seen = 0
    for slug, vec_dir, _, _ in corpora:
        shards = list_shards(vec_dir)
        if not shards:
            print(f"  WARN: no shards for {slug}, skipping", flush=True)
            counts[slug] = 0
            continue
        avail = sum(np.load(p, mmap_mode="r").shape[0] for p in shards)
        take_total = min(n_per_corpus, avail)
        seen_corpus = 0
        for shard_path in shards:
            if seen_corpus >= take_total:
                break
            arr = np.load(shard_path, mmap_mode="r")
            n_in_shard = arr.shape[0]
            take = min(n_in_shard, take_total - seen_corpus)
            for s in range(0, take, batch_size):
                e = min(s + batch_size, take)
                batch = torch.from_numpy(
                    np.ascontiguousarray(arr[s:e], dtype=np.float32)
                ).to(device)
                out = sae(batch)
                acts = out.latent_acts.cpu().numpy()
                idxs = out.latent_indices.cpu().numpy()
                chunk_base = seen_corpus + s
                n = acts.shape[0]
                for t in range(n):
                    ci = chunk_base + t
                    for j in range(acts.shape[1]):
                        a = float(acts[t, j])
                        if a <= 0:
                            continue
                        fid = int(idxs[t, j])
                        h = feat_heaps[fid]
                        # Tuple ordering: (act, slug, ci) — slug as tiebreaker
                        if len(h) < top_n:
                            heapq.heappush(h, (a, slug, ci))
                        elif a > h[0][0]:
                            heapq.heapreplace(h, (a, slug, ci))
            seen_corpus += take
        counts[slug] = seen_corpus
        total_seen += seen_corpus
        elapsed = time.monotonic() - t0
        rate = total_seen / max(elapsed, 1e-6)
        print(f"  [{slug:>22}] {seen_corpus:>7,} chunks  "
              f"(cum {total_seen:>9,}, {rate:.0f} ch/s)", flush=True)
    return feat_heaps, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--n-per-corpus", type=int, default=20000)
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    candidates = sorted(Path(p) for p in glob(args.sae_dir))
    if not candidates:
        raise FileNotFoundError(f"no matches for {args.sae_dir}")
    run_dir = candidates[0]
    ckpt = choose_checkpoint(run_dir)
    print(f"run: {run_dir.name}")
    print(f"checkpoint: {ckpt.name}")

    sae = Sae.load_from_disk(ckpt, device=args.device)
    print(f"SAE: {sae.cfg.sae_type.value}, num_latents={sae.num_latents}")

    print(f"scanning {len(CORPORA)} corpora @ {args.n_per_corpus:,} chunks each")
    feat_heaps, counts = collect_top_xlingual(
        sae, CORPORA, args.n_per_corpus, args.top_n,
        args.batch_size, args.device,
    )

    # Lazy-load chunk texts only for the corpora that produced top hits
    needed: dict[str, set[int]] = {}
    for fid, heap in enumerate(feat_heaps):
        for _, slug, ci in heap:
            needed.setdefault(slug, set()).add(ci)

    print(f"loading chunk texts for {len(needed)} corpora...")
    text_by_slug: dict[str, list[str]] = {}
    for slug, vec_dir, parquet_dir, glob_pat in CORPORA:
        if slug not in needed:
            continue
        max_idx = max(needed[slug]) + 1
        text_by_slug[slug] = load_chunk_texts(parquet_dir, glob_pat, max_idx)
        print(f"  {slug}: loaded {len(text_by_slug[slug]):,} texts")

    print("serializing...")
    live_features = []
    per_feature = {}
    for fid, heap in enumerate(feat_heaps):
        if not heap:
            continue
        sorted_hits = sorted(heap, key=lambda x: -x[0])
        entries = []
        for act, slug, ci in sorted_hits:
            texts = text_by_slug.get(slug, [])
            text = texts[ci] if ci < len(texts) else ""
            entries.append({
                "activation": round(act, 4),
                "corpus": slug,
                "chunk_idx": ci,
                "text": text,
                "window": text,
            })
        per_feature[str(fid)] = entries
        live_features.append(fid)

    out_path = Path(args.out) if args.out else run_dir / "feature_activations_xlingual.json"
    payload = {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "sae_type": sae.cfg.sae_type.value,
        "num_latents": sae.num_latents,
        "top_n_per_feature": args.top_n,
        "n_live_features": len(live_features),
        "live_feature_ids": live_features,
        "corpora": counts,
        "embedding_unit": "chunk",
        "features": per_feature,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    print(f"live features: {len(live_features)} / {sae.num_latents} "
          f"({len(live_features)/sae.num_latents:.1%})")


if __name__ == "__main__":
    main()
