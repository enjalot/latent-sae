"""Per-feature top-activating token windows for a ColBERT SAE checkpoint.

For each live feature, gathers the N tokens (across many chunks) that
activate it most strongly, along with a window of surrounding context
from the source parquet. Output is a JSON artifact ready for LLM labeling.

The source token embeddings are the SAME memmaps used in training
(`/data/embeddings/<dataset>-mxbai-edge-32m/train/`). We do NOT re-embed
— we just feed them through the SAE encoder.

Usage:
    python -m experiments.extract_feature_activations \\
        --sae-dir experiments/results/colbert_mxbai_phase4__k8_expansion_factor2_* \\
        --dataset fineweb \\
        --n-chunks 10000 \\
        --top-n 16
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


DATASET_PATHS = {
    "fineweb": {
        "vectors": "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train",
        "parquets": "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
    },
    "redpajama": {
        "vectors": "/data/embeddings/RedPajama-Data-V2-sample-10B-chunked-500-mxbai-edge-32m/train",
        "parquets": "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-500/train",
    },
    "pile": {
        "vectors": "/data/embeddings/pile-uncopyrighted-chunked-500-mxbai-edge-32m/train",
        "parquets": "/data/chunks/pile-uncopyrighted-chunked-500/train",
    },
}

TOKENIZER_ID = "bert-base-multilingual-cased"


def load_chunk_texts(parquet_dir: str, n_chunks: int) -> list[str]:
    """Read chunk_text in parquet order until we have n_chunks rows."""
    parquets = sorted(Path(parquet_dir).glob("data-*.parquet"))
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
def collect_top_activations(sae: Sae, vectors: np.ndarray, offsets: np.ndarray,
                            n_chunks: int, top_n: int, batch_size: int = 4096,
                            device: str = "cuda") -> list[list[tuple]]:
    """For each latent, return its top-N (activation, chunk_idx, token_idx) heap.

    Streams tokens through SAE; maintains a min-heap per feature of size top_n.
    """
    sae.eval()
    num_latents = sae.num_latents
    # Min-heaps of (act, chunk_idx, token_idx) per feature
    feat_heaps: list[list[tuple]] = [[] for _ in range(num_latents)]

    total_tokens = int(offsets[n_chunks])
    print(f"  streaming {total_tokens:,} tokens across {n_chunks:,} chunks...")

    # Precompute per-token (chunk_idx, token_idx) for the first n_chunks
    chunk_of_token = np.empty(total_tokens, dtype=np.int32)
    token_of_token = np.empty(total_tokens, dtype=np.int32)
    for c in range(n_chunks):
        s, e = int(offsets[c]), int(offsets[c + 1])
        chunk_of_token[s:e] = c
        token_of_token[s:e] = np.arange(e - s, dtype=np.int32)

    t0 = time.monotonic()
    for s in range(0, total_tokens, batch_size):
        e = min(s + batch_size, total_tokens)
        batch = torch.from_numpy(
            np.ascontiguousarray(vectors[s:e], dtype=np.float32)
        ).to(device)
        out = sae(batch)
        acts = out.latent_acts         # (B, k)
        idxs = out.latent_indices      # (B, k)
        # Flatten
        acts_np = acts.cpu().numpy()
        idxs_np = idxs.cpu().numpy()
        batch_chunks = chunk_of_token[s:e]
        batch_tokens = token_of_token[s:e]
        # For each token in batch, push its (k) active features into heaps
        n = acts_np.shape[0]
        for t in range(n):
            c = int(batch_chunks[t])
            ti = int(batch_tokens[t])
            for j in range(acts_np.shape[1]):
                a = float(acts_np[t, j])
                if a <= 0:
                    continue
                fid = int(idxs_np[t, j])
                h = feat_heaps[fid]
                if len(h) < top_n:
                    heapq.heappush(h, (a, c, ti))
                elif a > h[0][0]:
                    heapq.heapreplace(h, (a, c, ti))
    print(f"  streamed in {time.monotonic() - t0:.1f}s")
    return feat_heaps


def render_window(tokens: list[str], pos: int, radius: int = 10) -> str:
    lo = max(0, pos - radius)
    hi = min(len(tokens), pos + radius + 1)
    pieces = []
    for i in range(lo, hi):
        tok = tokens[i].replace("##", "")
        if i == pos:
            pieces.append(f"<<{tok}>>")
        else:
            pieces.append(tok)
    return " ".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--dataset", default="fineweb", choices=list(DATASET_PATHS))
    ap.add_argument("--n-chunks", type=int, default=10000)
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument("--out", default=None, help="output JSON path (default: run-dir/feature_activations.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    candidates = sorted(Path(p) for p in glob(args.sae_dir))
    if not candidates:
        raise FileNotFoundError(f"no matches for {args.sae_dir}")
    run_dir = candidates[0]
    ckpt = next((p for p in (run_dir / "checkpoints").glob("*")
                 if p.is_dir() and (p / "cfg.json").exists()), None)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    print(f"run: {run_dir.name}")
    print(f"checkpoint: {ckpt.name}")

    # Load SAE
    sae = Sae.load_from_disk(ckpt, device=args.device)
    print(f"SAE: {sae.cfg.sae_type.value}, num_latents={sae.num_latents}, d_in={sae.W_dec.shape[-1]}")

    # Load vectors + offsets for this dataset
    paths = DATASET_PATHS[args.dataset]
    vec_file = next(Path(paths["vectors"]).glob("data-*.npy"))
    vectors = np.load(vec_file, mmap_mode="r")
    offsets = np.load(Path(paths["vectors"]) / "chunk_offsets.npy")
    print(f"vectors: {vectors.shape}")
    print(f"chunks available: {len(offsets) - 1:,}")
    n_chunks = min(args.n_chunks, len(offsets) - 1)

    # Collect top activations
    feat_heaps = collect_top_activations(sae, vectors, offsets, n_chunks,
                                         args.top_n, args.batch_size, args.device)

    # Load chunk texts + tokenize for context windows
    print(f"loading {n_chunks:,} chunk texts...")
    chunk_texts = load_chunk_texts(paths["parquets"], n_chunks)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # For each live feature, serialize top activations with context windows
    print("rendering feature windows...")
    live_features = []
    tok_cache: dict[int, list[str]] = {}

    def get_tokens(ci: int) -> list[str]:
        if ci not in tok_cache:
            tok_cache[ci] = tokenizer.tokenize(chunk_texts[ci])
        return tok_cache[ci]

    per_feature = {}
    for fid, heap in enumerate(feat_heaps):
        if not heap:
            continue
        # Sort descending by activation
        sorted_hits = sorted(heap, key=lambda x: -x[0])
        entries = []
        for act, ci, ti in sorted_hits:
            tokens = get_tokens(ci)
            ti_safe = min(ti, len(tokens) - 1)
            entries.append({
                "activation": round(act, 4),
                "chunk_idx": ci,
                "token_idx": ti,
                "window": render_window(tokens, ti_safe, radius=10),
            })
        per_feature[str(fid)] = entries
        live_features.append(fid)

    out_path = Path(args.out) if args.out else run_dir / "feature_activations.json"
    payload = {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "sae_type": sae.cfg.sae_type.value,
        "num_latents": sae.num_latents,
        "n_chunks_scanned": n_chunks,
        "n_tokens_scanned": int(offsets[n_chunks]),
        "top_n_per_feature": args.top_n,
        "n_live_features": len(live_features),
        "live_feature_ids": live_features,
        "features": per_feature,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    print(f"live features: {len(live_features)} / {sae.num_latents}")


if __name__ == "__main__":
    main()
