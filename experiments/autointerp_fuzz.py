"""Fuzzing scorer (Neuronpedia/Delphi-style).

Tests position-specificity of feature descriptions. For each labeled
feature:

  - POSITIVE: take a true top-activating window. The activating token is
    already marked with <<token>> in the stored window. Ask the LLM
    whether the marked token is the one the description describes. Expect YES.
  - NEGATIVE: take the SAME window but re-mark a DIFFERENT, random token in
    the window (far from the activating position). Expect NO.

Balanced accuracy across both halves measures whether the description
narrows to the right token, not just the right context.

Distinct from `autointerp_predict.py` (detection), which uses different
windows for pos/neg. Fuzzing holds the context constant and only varies
the marker position.

Usage:
  python -m experiments.autointerp_fuzz \\
      --run-dir experiments/results/<run> \\
      --model qwen2.5:7b-instruct-q4_K_M
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MARKER_RE = re.compile(r"<<(.+?)>>")


FUZZ_TEMPLATE = """You are checking whether a sparse autoencoder feature should fire on a SPECIFIC token.

Feature description: "{description}"

Token context (the candidate token is marked with <<>>):
{window}

Is the MARKED token the one this feature activates on? Reply with exactly YES or NO — nothing else."""


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


def split_window_tokens(window: str) -> tuple[list[str], int]:
    """Parse a <<token>>-marked window into (tokens, marked_pos).
    marked_pos is -1 if no marker present."""
    # Tokens are whitespace-separated (that's how render_window builds them).
    tokens = window.split()
    marked_pos = -1
    for i, t in enumerate(tokens):
        m = MARKER_RE.match(t)
        if m:
            tokens[i] = m.group(1)
            marked_pos = i
            break
    return tokens, marked_pos


def render_marked(tokens: list[str], pos: int) -> str:
    return " ".join(f"<<{t}>>" if i == pos else t for i, t in enumerate(tokens))


def pick_negative_position(n_tokens: int, pos_tok: int, rng: random.Random,
                           min_distance: int = 4) -> int | None:
    """Pick a different position that is at least min_distance away."""
    candidates = [i for i in range(n_tokens) if abs(i - pos_tok) >= min_distance]
    if not candidates:
        return None
    return rng.choice(candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--n-features", type=int, default=0,
                    help="0 = score every labeled feature")
    ap.add_argument("--grade-all-labels", default="COHERENT,THEMATIC",
                    help="Comma-separated label classes to fully grade. Other "
                         "classes are stratified-sampled to fill --n-features")
    ap.add_argument("--pairs-per-feature", type=int, default=3,
                    help="Number of (positive, negative) pairs per feature")
    ap.add_argument("--desc-windows-seen", type=int, default=10)
    ap.add_argument("--min-distance", type=int, default=4,
                    help="Negative token must be this far from positive token")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    labels = json.loads((run_dir / "feature_labels.json").read_text())
    acts_data = json.loads((run_dir / "feature_activations.json").read_text())

    feats_by_label: dict[str, list[str]] = {}
    for fid, rec in labels["labels"].items():
        feats_by_label.setdefault(rec["judgment"], []).append(fid)
    order = ["COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"]
    rng = random.Random(args.seed)

    sampled: list[tuple[str, str]] = []
    grade_all = {s.strip().upper() for s in args.grade_all_labels.split(",") if s.strip()}
    for lbl in order:
        if lbl in grade_all:
            bucket = feats_by_label.get(lbl, []).copy()
            rng.shuffle(bucket)
            sampled.extend([(fid, lbl) for fid in bucket])
    other = [l for l in order if l not in grade_all and feats_by_label.get(l)]
    if args.n_features > 0 and other:
        per = max(1, args.n_features // len(other))
        for lbl in other:
            bucket = feats_by_label.get(lbl, []).copy()
            rng.shuffle(bucket)
            sampled.extend([(fid, lbl) for fid in bucket[:per]])
    elif args.n_features == 0:
        # also fully grade the others
        for lbl in other:
            bucket = feats_by_label.get(lbl, []).copy()
            rng.shuffle(bucket)
            sampled.extend([(fid, lbl) for fid in bucket])

    print(f"fuzzing {len(sampled)} features ({sum(1 for _, l in sampled if l in grade_all)} fully graded, "
          f"{len(sampled) - sum(1 for _, l in sampled if l in grade_all)} sampled)")

    per_feature_results = []
    t0 = time.monotonic()
    total_correct = 0
    total_n = 0

    for fi, (fid, lbl) in enumerate(sampled):
        description = labels["labels"][fid]["description"]
        hits = acts_data["features"][fid]
        # Use held-out positives (beyond those the labeler saw)
        held = hits[args.desc_windows_seen:] or hits[-args.pairs_per_feature:]
        rng.shuffle(held)
        pool = held[: args.pairs_per_feature]

        pos_correct = 0
        neg_correct = 0
        n_pos = 0
        n_neg = 0
        details = []

        for h in pool:
            window = h.get("window", "")
            tokens, pos_tok = split_window_tokens(window)
            if pos_tok < 0 or not tokens:
                continue

            # POSITIVE: re-render same tokens with same marker (normalizes spacing)
            pos_prompt = FUZZ_TEMPLATE.format(description=description,
                                              window=render_marked(tokens, pos_tok))
            pred_p = ollama_yesno(args.model, pos_prompt)
            if pred_p is not None:
                n_pos += 1
                if pred_p is True:
                    pos_correct += 1

            # NEGATIVE: pick a distant token, re-mark
            neg_i = pick_negative_position(len(tokens), pos_tok, rng,
                                           min_distance=args.min_distance)
            if neg_i is None:
                continue
            neg_prompt = FUZZ_TEMPLATE.format(description=description,
                                              window=render_marked(tokens, neg_i))
            pred_n = ollama_yesno(args.model, neg_prompt)
            if pred_n is not None:
                n_neg += 1
                if pred_n is False:
                    neg_correct += 1

            details.append({
                "pos_token": tokens[pos_tok], "neg_token": tokens[neg_i],
                "pred_pos_yes": pred_p, "pred_neg_yes": pred_n,
            })

        # Balanced accuracy (mean of per-class recall)
        pos_rec = pos_correct / n_pos if n_pos else 0.0
        neg_rec = neg_correct / n_neg if n_neg else 0.0
        bal_acc = 0.5 * (pos_rec + neg_rec) if (n_pos and n_neg) else 0.0

        per_feature_results.append({
            "feature": fid, "label": lbl,
            "description": description[:200],
            "n_pos": n_pos, "n_neg": n_neg,
            "pos_recall": pos_rec, "neg_recall": neg_rec,
            "bal_acc": bal_acc,
            "details": details,
        })
        total_correct += (pos_correct + neg_correct)
        total_n += (n_pos + n_neg)

        elapsed = time.monotonic() - t0
        rate = (fi + 1) / elapsed if elapsed > 0 else 0
        eta = (len(sampled) - fi - 1) / rate if rate > 0 else float("inf")
        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"  [{fi+1:3d}/{len(sampled)}] f{fid} ({lbl}) "
                  f"bal_acc={bal_acc:.2f} ({rate:.1f}/s, eta {eta/60:.1f}m)")

    by_label = {}
    for r in per_feature_results:
        by_label.setdefault(r["label"], []).append(r["bal_acc"])

    report = {
        "run": run_dir.name,
        "scorer": "fuzzing",
        "model": args.model,
        "n_features_scored": len(per_feature_results),
        "overall_bal_acc": float(np.mean([r["bal_acc"] for r in per_feature_results])) if per_feature_results else 0.0,
        "overall_raw_acc": total_correct / max(total_n, 1),
        "per_label_bal_acc": {k: float(np.mean(v)) for k, v in by_label.items()},
        "per_label_n": {k: len(v) for k, v in by_label.items()},
        "per_feature": per_feature_results,
    }
    out = run_dir / "feature_fuzz_scores.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\noverall balanced accuracy: {report['overall_bal_acc']:.3f}  "
          f"raw acc: {report['overall_raw_acc']:.3f}")
    for lbl in order:
        if lbl in by_label:
            print(f"  {lbl:<13} n={len(by_label[lbl]):3d}  bal_acc={np.mean(by_label[lbl]):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
