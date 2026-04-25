"""Domain-probe cross-SAE comparison.

Pre-defines a dozen text domains (e.g., code, legal, medical, math…), finds
source chunks most similar to each domain's seed text, runs those chunks
through each SAE checkpoint, and reports the top-K features firing per
domain. Gives a consistent fingerprint for comparing SAEs at the concept
level — independent of proxy metrics or retrieval.

For each SAE: domain → list of (feature_id, activation_sum_over_tokens,
n_chunks_featured_in).

Usage:
    python -m experiments.domain_probe --sae-dir <run_dir> [<run_dir>...] \\
        --dataset fineweb --n-chunks-per-domain 20 --top-k-features 12
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

MODEL_ID = "mixedbread-ai/mxbai-edge-colbert-v0-32m"

DATASET_PATHS = {
    "fineweb": {
        "vectors": "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train",
        "parquets": "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
    },
}

DOMAINS = {
    "code_python": "def fibonacci(n): if n < 2: return n; return fibonacci(n-1) + fibonacci(n-2). class LinkedList: def __init__(self): self.head = None.",
    "legal": "Pursuant to Article 12 of the contract, the parties agree to binding arbitration in the jurisdiction of Delaware. The plaintiff shall provide written notice.",
    "medical_clinical": "The patient presented with acute myocardial infarction. ECG showed ST-segment elevation in leads V1-V4. Troponin I was elevated at 5.2 ng/mL.",
    "math_proof": "Let f be a continuous function on [a,b]. By the Mean Value Theorem, there exists c in (a,b) such that f'(c) = (f(b) - f(a)) / (b - a).",
    "chemistry": "The reaction of sodium hydroxide with hydrochloric acid produces sodium chloride and water: NaOH + HCl -> NaCl + H2O. The pKa of acetic acid is 4.76.",
    "recipes_cooking": "Preheat the oven to 375 degrees Fahrenheit. In a large bowl, whisk together the flour, baking soda, and salt. Gradually fold in the chocolate chips.",
    "sports_commentary": "And he scores! A stunning goal from outside the box in the 89th minute! The striker rounds the goalkeeper and slots it into the empty net.",
    "historical_narrative": "In the summer of 1215, King John of England met the rebel barons at Runnymede and affixed his seal to the Magna Carta, a charter of liberties.",
    "poetry": "Shall I compare thee to a summer's day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May, and summer's lease hath all too short a date.",
    "dialogue_conversation": "\"What time is it?\" she asked. \"Nearly midnight,\" he replied with a sigh. \"We should head home before it gets any later.\" She nodded in silent agreement.",
    "product_reviews": "Five stars! This vacuum cleaner is incredibly powerful and quiet. The cordless design makes it easy to maneuver around furniture. Highly recommend for pet owners.",
    "scientific_abstract": "We investigated the effect of temperature on enzyme kinetics. Results show that reaction rate doubles between 20°C and 30°C, consistent with Q10 theory.",
}


# ---------- Domain chunk finder ----------

def find_domain_chunks(domain_text: str, chunk_texts: list[str],
                       n: int = 20) -> list[int]:
    """Use TF-IDF cosine similarity to find the n most similar chunks."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(stop_words="english", max_features=20_000,
                          ngram_range=(1, 2))
    X = vec.fit_transform(chunk_texts + [domain_text])
    sims = cosine_similarity(X[-1:], X[:-1]).ravel()
    top = np.argsort(-sims)[:n]
    return [int(i) for i in top]


# ---------- Per-SAE feature activation aggregator ----------

@torch.no_grad()
def feature_sums_for_chunks(sae: Sae, vectors: np.ndarray, offsets: np.ndarray,
                            chunk_idxs: list[int], batch_size: int = 4096,
                            device: str = "cuda") -> np.ndarray:
    """Return (num_latents,) sum of activations across all tokens of the chunks."""
    sae.eval()
    num_latents = sae.num_latents
    sums = torch.zeros(num_latents, device=device)
    counts = torch.zeros(num_latents, device=device, dtype=torch.int64)

    for ci in chunk_idxs:
        s, e = int(offsets[ci]), int(offsets[ci + 1])
        if e <= s:
            continue
        batch = torch.from_numpy(
            np.ascontiguousarray(vectors[s:e], dtype=np.float32)
        ).to(device)
        out = sae(batch)
        acts = out.latent_acts          # (nt, k)
        idxs = out.latent_indices       # (nt, k)
        mask = acts > 0
        idxs_flat = idxs[mask].long()
        acts_flat = acts[mask]
        sums.scatter_add_(0, idxs_flat, acts_flat)
        # Count unique chunks each feature fires in
        fired_any_in_chunk = torch.zeros(num_latents, device=device,
                                          dtype=torch.bool)
        fired_any_in_chunk.scatter_(0, idxs_flat, torch.ones_like(idxs_flat,
                                                                   dtype=torch.bool))
        counts += fired_any_in_chunk.long()

    return sums.cpu().numpy(), counts.cpu().numpy()


def load_chunk_texts(parquet_dir: str, n: int) -> list[str]:
    parquets = sorted(Path(parquet_dir).glob("data-*.parquet"))
    frames, got = [], 0
    for p in parquets:
        df = pd.read_parquet(p, columns=["chunk_text"])
        frames.append(df)
        got += len(df)
        if got >= n:
            break
    return pd.concat(frames, ignore_index=True).head(n)["chunk_text"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", action="append", default=[])
    ap.add_argument("--dataset", default="fineweb")
    ap.add_argument("--n-chunks-per-domain", type=int, default=20)
    ap.add_argument("--top-k-features", type=int, default=12)
    ap.add_argument("--n-candidate-chunks", type=int, default=20000,
                    help="pool of chunks to search over per domain")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",
                    default="/data/embeddings/beir/domain_probe.json")
    args = ap.parse_args()

    # Resolve run dirs
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

    # Step 1: find chunks per domain (once, SAE-independent)
    domain_chunks: dict[str, list[int]] = {}
    print("finding per-domain chunks...")
    for name, seed in DOMAINS.items():
        idxs = find_domain_chunks(seed, chunk_texts, n=args.n_chunks_per_domain)
        # Only keep chunks whose index is within our embedded set
        idxs = [i for i in idxs if i < len(offsets) - 1]
        domain_chunks[name] = idxs
        print(f"  {name:<22} {len(idxs)} chunks  (first 3 previews):")
        for ci in idxs[:3]:
            preview = chunk_texts[ci][:80].replace("\n", " ")
            print(f"    ci={ci:>5} {preview!r}")

    # Step 2: for each SAE, per-domain top features
    results = {"domains": domain_chunks, "saes": {}}
    for rd in run_dirs:
        ckpt = next((p for p in (rd / "checkpoints").glob("*")
                     if p.is_dir() and (p / "cfg.json").exists()), None)
        if ckpt is None:
            print(f"SKIP {rd.name}: no checkpoint")
            continue
        print(f"\n== {rd.name} ==")
        sae = Sae.load_from_disk(ckpt, device=args.device)
        per_sae: dict[str, list[dict]] = {}
        for name, idxs in domain_chunks.items():
            t = time.monotonic()
            sums, counts = feature_sums_for_chunks(sae, vectors, offsets, idxs,
                                                   device=args.device)
            # Rank by sum; tiebreak by count
            order = np.argsort(-sums)[: args.top_k_features]
            per_sae[name] = [
                {"feature": int(fid), "act_sum": float(sums[fid]),
                 "n_chunks_with_activation": int(counts[fid])}
                for fid in order if sums[fid] > 0
            ]
            print(f"  {name:<22} top-{args.top_k_features}: "
                  f"{[h['feature'] for h in per_sae[name]]}  "
                  f"({time.monotonic() - t:.2f}s)")
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
