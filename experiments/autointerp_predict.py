"""Autointerp prediction score (SAEBench-style).

Consumes feature_labels.json (descriptions from label_features_ollama) and
feature_activations.json (ground truth: top-N activating token windows plus
held-out sample of non-activating windows). For each feature:

  1. Show the LLM the description and a new token window.
  2. Ask: "Does this feature fire on this token? YES or NO"
  3. Score accuracy across (held-out activating tokens) + (zero-activation tokens).

Aggregate per SAE = mean per-feature prediction accuracy.

Usage:
    python -m experiments.autointerp_predict \\
        --run-dir experiments/results/<run> \\
        --dataset fineweb --model qwen2.5:7b-instruct-q4_K_M
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TOKENIZER_ID = "bert-base-multilingual-cased"

DATASET_PATHS = {
    "fineweb": {
        "vectors": "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-mxbai-edge-32m/train",
        "parquets": "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train",
    },
}


PREDICT_TEMPLATE = """You are checking whether a sparse autoencoder feature should fire on a new token.

Feature description: "{description}"

Token context (the candidate token is marked with <<>>):
{window}

Based on the description, would this feature activate on the marked token? Reply with exactly YES or NO — nothing else."""


def ollama_yesno(model: str, prompt: str) -> bool | None:
    r = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 4, "num_ctx": 2048},
    }, timeout=120)
    r.raise_for_status()
    out = r.json()["response"].strip().upper()
    if out.startswith("YES"):
        return True
    if out.startswith("NO"):
        return False
    return None


@torch.no_grad()
def per_feature_activations(sae: Sae, vectors: np.ndarray, token_idxs: list[int],
                            device: str = "cuda") -> np.ndarray:
    """For a list of global token indices, return (n_tokens, num_latents) sparse activations."""
    sae.eval()
    n = len(token_idxs)
    batch = torch.from_numpy(
        np.ascontiguousarray(vectors[token_idxs], dtype=np.float32)
    ).to(device)
    out = sae(batch)
    acts = out.latent_acts.cpu().numpy()       # (n, k)
    idxs = out.latent_indices.cpu().numpy()    # (n, k)
    dense = np.zeros((n, sae.num_latents), dtype=np.float32)
    for i in range(n):
        dense[i, idxs[i]] = acts[i]
    return dense


def render_window(tokens: list[str], pos: int, radius: int = 10) -> str:
    lo = max(0, pos - radius)
    hi = min(len(tokens), pos + radius + 1)
    pieces = []
    for i in range(lo, hi):
        tok = tokens[i].replace("##", "")
        pieces.append(f"<<{tok}>>" if i == pos else tok)
    return " ".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="fineweb", choices=list(DATASET_PATHS))
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--n-features", type=int, default=64,
                    help="Sample this many labeled features to score")
    ap.add_argument("--grade-all-labels", default=None,
                    help="Comma-separated label classes to grade in their entirety (e.g. 'COHERENT,THEMATIC'). Overrides --n-features for those; stratified sample fills the rest.")
    ap.add_argument("--pos-per-feature", type=int, default=3,
                    help="Held-out positive (high-activation) tokens per feature")
    ap.add_argument("--neg-per-feature", type=int, default=3,
                    help="Negative (zero-activation) tokens per feature")
    ap.add_argument("--desc-windows-seen", type=int, default=10,
                    help="How many top windows the labeler showed the LLM; positives start after this index.")
    ap.add_argument("--chunks-for-negatives", type=int, default=500,
                    help="Sample negatives from this many chunks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    labels = json.loads((run_dir / "feature_labels.json").read_text())
    acts_data = json.loads((run_dir / "feature_activations.json").read_text())

    # Load SAE + source vectors
    ckpt = next((p for p in (run_dir / "checkpoints").glob("*")
                 if p.is_dir() and (p / "cfg.json").exists()), None)
    if ckpt is None:
        raise FileNotFoundError(f"no SAE checkpoint in {run_dir}")
    sae = Sae.load_from_disk(ckpt, device=args.device)

    paths = DATASET_PATHS[args.dataset]
    vec_file = next(Path(paths["vectors"]).glob("data-*.npy"))
    vectors = np.load(vec_file, mmap_mode="r")
    offsets = np.load(Path(paths["vectors"]) / "chunk_offsets.npy")

    # Pick features to score. Prefer coherent/thematic because we expect them
    # to be predictable; include a handful of others for balance.
    feats_by_label: dict[str, list[str]] = {"COHERENT": [], "THEMATIC": [],
                                             "GENERIC": [], "POLYSEMANTIC": [],
                                             "UNCLEAR": []}
    for fid, rec in labels["labels"].items():
        feats_by_label.setdefault(rec["judgment"], []).append(fid)

    rng = random.Random(args.seed)
    sampled = []
    order = ["COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"]
    for lbl in order:
        bucket = feats_by_label.get(lbl, [])
        rng.shuffle(bucket)

    if args.grade_all_labels:
        # Include EVERY feature in named labels; stratified sample from the rest
        grade_all = {s.strip().upper() for s in args.grade_all_labels.split(",") if s.strip()}
        for lbl in order:
            if lbl in grade_all:
                sampled.extend([(fid, lbl) for fid in feats_by_label[lbl]])
        # Fill remainder of --n-features from the non-all classes, evenly
        other = [l for l in order if l not in grade_all and feats_by_label[l]]
        if other and args.n_features > 0:
            per_class = max(1, args.n_features // len(other))
            for lbl in other:
                sampled.extend([(fid, lbl) for fid in feats_by_label[lbl][:per_class]])
    else:
        remaining = args.n_features
        for lbl in order:
            take = min(len(feats_by_label[lbl]),
                       max(1, remaining // max(1, len([l for l in order
                                                        if feats_by_label[l]]))))
            sampled.extend([(fid, lbl) for fid in feats_by_label[lbl][:take]])
            remaining = args.n_features - len(sampled)
            if remaining <= 0:
                break
        sampled = sampled[: args.n_features]

    # Load chunk texts + tokenizer for rendering windows
    parquets = sorted(Path(paths["parquets"]).glob("data-*.parquet"))
    # Cover enough chunks to include all referenced chunk_idxs from top windows
    # plus negatives (from the first N chunks we scan).
    needed_chunks = set()
    for fid, _ in sampled:
        for h in acts_data["features"][fid]:
            needed_chunks.add(h["chunk_idx"])
    max_chunk = max(needed_chunks) if needed_chunks else 0
    n_text_chunks = max(max_chunk + 1, args.chunks_for_negatives)
    frames, got = [], 0
    for p in parquets:
        df = pd.read_parquet(p, columns=["chunk_text"])
        frames.append(df)
        got += len(df)
        if got >= n_text_chunks:
            break
    chunk_texts = pd.concat(frames, ignore_index=True)["chunk_text"].head(n_text_chunks).tolist()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # Choose negative candidate tokens: pick `chunks_for_negatives` random chunks
    # and random token positions within them. Score against SAE to ensure those
    # tokens have zero activation for the target feature.
    neg_candidate_tokens: list[int] = []  # global token indices
    rng_neg = random.Random(args.seed + 1)
    neg_chunks = rng_neg.sample(range(min(args.chunks_for_negatives,
                                          len(offsets) - 1)),
                                 k=min(args.chunks_for_negatives,
                                       len(offsets) - 1))
    for ci in neg_chunks:
        cs, ce = int(offsets[ci]), int(offsets[ci + 1])
        neg_candidate_tokens.extend(range(cs, ce))
    rng_neg.shuffle(neg_candidate_tokens)

    # Pre-score all candidate negatives once (avoids per-feature GPU call)
    print(f"pre-scoring {len(neg_candidate_tokens):,} candidate negative tokens...")
    neg_sample = neg_candidate_tokens[: min(10_000, len(neg_candidate_tokens))]
    neg_acts = per_feature_activations(sae, vectors, neg_sample, device=args.device)
    neg_per_feature = {}  # fid → list of global token idx that have 0 activation for that feature

    per_feature_results = []
    total_correct = 0
    total_n = 0
    t0 = time.monotonic()

    for fi, (fid, lbl) in enumerate(sampled):
        fid_i = int(fid)
        description = labels["labels"][fid]["description"]
        hits = acts_data["features"][fid]

        # Positives: hits beyond the windows shown to the description LLM
        # (truly held-out). Falls back to the last N hits if that set is empty.
        holdout = hits[args.desc_windows_seen:]
        if len(holdout) >= args.pos_per_feature:
            pos_sel = holdout[: args.pos_per_feature]
        else:
            pos_sel = hits[-args.pos_per_feature:]

        # Negatives: first k tokens in neg_sample with zero act for this feature
        zero_mask = neg_acts[:, fid_i] == 0
        zero_global = [neg_sample[i] for i, z in enumerate(zero_mask) if z]
        if len(zero_global) < args.neg_per_feature:
            print(f"  WARN feature {fid}: only {len(zero_global)} zero-activation negatives")
        neg_sel = zero_global[: args.neg_per_feature]

        # Build prompts
        examples = []
        for h in pos_sel:
            examples.append(("POS", h["chunk_idx"], h["token_idx"],
                             h.get("window", "")))
        for g_idx in neg_sel:
            ci = int(np.searchsorted(offsets, g_idx + 1)) - 1
            ti = int(g_idx - offsets[ci])
            # Tokenize chunk to render window
            tokens = tokenizer.tokenize(chunk_texts[ci])
            ti_safe = min(ti, len(tokens) - 1) if tokens else 0
            window = render_window(tokens, ti_safe, radius=10) if tokens else ""
            examples.append(("NEG", ci, ti, window))

        # Score
        correct = 0
        details = []
        for kind, ci, ti, window in examples:
            if not window:
                continue
            pred = ollama_yesno(args.model,
                                PREDICT_TEMPLATE.format(description=description,
                                                        window=window))
            if pred is None:
                continue
            truth = (kind == "POS")
            if pred == truth:
                correct += 1
            details.append({"kind": kind, "pred_yes": pred, "window": window[:120]})
        n = sum(1 for d in details)
        acc = correct / n if n else 0.0
        per_feature_results.append({
            "feature": fid, "label": lbl,
            "description": description[:200],
            "accuracy": acc, "n": n, "details": details,
        })
        total_correct += correct
        total_n += n

        elapsed = time.monotonic() - t0
        rate = (fi + 1) / elapsed if elapsed > 0 else 0
        eta = (len(sampled) - fi - 1) / rate if rate > 0 else float("inf")
        print(f"  [{fi+1:3d}/{len(sampled)}] f{fid} ({lbl}) acc={acc:.2f}  "
              f"({rate:.1f}/s, eta {eta/60:.1f}m)")

    # Aggregate per-label
    by_label = {}
    for r in per_feature_results:
        by_label.setdefault(r["label"], []).append(r["accuracy"])

    report = {
        "run": run_dir.name,
        "model": args.model,
        "n_features_scored": len(per_feature_results),
        "overall_accuracy": total_correct / max(total_n, 1),
        "per_label_accuracy": {k: float(np.mean(v)) for k, v in by_label.items()},
        "per_label_n": {k: len(v) for k, v in by_label.items()},
        "per_feature": per_feature_results,
    }
    out_path = run_dir / "feature_prediction_scores.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\noverall accuracy: {report['overall_accuracy']:.3f}")
    for lbl, acc in report["per_label_accuracy"].items():
        print(f"  {lbl:<13} n={report['per_label_n'][lbl]:3d}  acc={acc:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
