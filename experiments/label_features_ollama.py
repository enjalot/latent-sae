"""Label SAE features via Ollama (qwen2.5:32b) autointerp.

Consumes a feature_activations.json produced by extract_feature_activations.py
and asks a local Ollama model to write a 1-2 sentence description for each
live feature based on the top-activating token windows.

Also computes a simple interpretability score: for each feature, ask the
model to flag whether the described concept is "clearly coherent" vs
"vague / polysemantic / noise". Aggregate statistics let us compare
SAE configs by percentage of clearly coherent features.

Usage:
    python -m experiments.label_features_ollama \\
        --input experiments/results/<run>/feature_activations.json \\
        --model qwen2.5:32b-instruct-q4_K_M \\
        --max-features 128 --max-windows 10
"""
import argparse
import json
import time
from pathlib import Path

import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

DESCRIBE_TEMPLATE = """You are analyzing features from a sparse autoencoder trained on ColBERT per-token embeddings from web text.

Below are the {n} contexts (10-token windows) where feature {fid} activates most strongly. The activating token is marked with <<>>. Higher activation = stronger fire.

{windows}

In 1-2 short sentences, describe what concept or pattern this feature responds to. Focus on the activating token's role and what the surrounding contexts have in common. If the contexts look unrelated or the activating tokens are all punctuation / function words, say so bluntly.

Description:"""


POSITIONAL_DESCRIBE_TEMPLATE = """You are analyzing features from a sparse autoencoder trained on ColBERT per-token embeddings from web text.

Below are the {n} contexts (10-token windows) where feature {fid} activates most strongly. The activating token is marked with <<>>. Higher activation = stronger fire.

{windows}

Your job is to describe the specific TOKEN-LEVEL pattern that makes the MARKED token fire — NOT the general topic of the surrounding text. Be precise: what syntactic role, morphological form, semantic category, or discourse position does the activating token occupy?

Crucial specificity test: if other tokens in the same context could also plausibly match your description (so a reader couldn't pick out which one was marked), your description is insufficient — state that.

In 1-2 short sentences, describe the feature in terms of the activating TOKEN. If the marked tokens look unrelated across contexts, or are all punctuation / function words, say so bluntly.

Description:"""


JUDGE_TEMPLATE = """You wrote this description of a sparse autoencoder feature:

"{description}"

Classify the feature into exactly ONE of these categories:
- COHERENT — clear, specific concept shared across examples (e.g., "Python list comprehensions", "medical diagnosis terms", "currency symbols")
- THEMATIC — loose topical cluster without specific token-level pattern (e.g., "scientific writing", "news text")
- GENERIC — punctuation, function words, sentence-position markers, or generic markers of structure
- POLYSEMANTIC — activations span multiple unrelated concepts
- UNCLEAR — description is too vague to judge

Reply with exactly one word: COHERENT, THEMATIC, GENERIC, POLYSEMANTIC, or UNCLEAR."""


def ollama_generate(model: str, prompt: str, max_tokens: int = 180,
                    temperature: float = 0.2) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 4096,
        },
    }, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip()


def label_one(feature_id: str, windows: list[dict], model: str,
              max_windows: int = 10, prompt_variant: str = "default") -> dict:
    sel = windows[:max_windows]
    formatted = "\n".join(
        f"{i+1}. [act={w['activation']:.2f}] {w['window']}"
        for i, w in enumerate(sel)
    )
    tmpl = POSITIONAL_DESCRIBE_TEMPLATE if prompt_variant == "positional" else DESCRIBE_TEMPLATE
    prompt = tmpl.format(n=len(sel), fid=feature_id, windows=formatted)
    desc = ollama_generate(model, prompt, max_tokens=200, temperature=0.2)

    judge_prompt = JUDGE_TEMPLATE.format(description=desc)
    raw = ollama_generate(model, judge_prompt, max_tokens=10, temperature=0.0)
    label = raw.split()[0].strip().upper() if raw else "UNCLEAR"
    # Defensive normalization
    valid = {"COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"}
    if label not in valid:
        label = "UNCLEAR"
    return {"description": desc, "judgment": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to feature_activations.json")
    ap.add_argument("--model", default="qwen2.5:32b-instruct-q4_K_M")
    ap.add_argument("--max-windows", type=int, default=10,
                    help="top-N windows per feature to pass to LLM")
    ap.add_argument("--max-features", type=int, default=None,
                    help="cap on features labeled (None = all live)")
    ap.add_argument("--sample-random", type=int, default=None,
                    help="random-sample this many features (seeded). Takes precedence over --max-features.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="output path (default: <input_dir>/feature_labels.json or _sample<N>.json if sampled)")
    ap.add_argument("--resume", action="store_true",
                    help="skip features already in output file")
    ap.add_argument("--prompt-variant", default="default",
                    choices=["default", "positional"],
                    help="which DESCRIBE prompt to use. 'positional' targets the "
                         "ACTIVATING TOKEN specifically, vs 'default' which is "
                         "context-level.")
    args = ap.parse_args()

    in_path = Path(args.input)
    suffix = "_positional" if args.prompt_variant == "positional" else ""
    if args.out:
        out_path = Path(args.out)
    elif args.sample_random:
        out_path = in_path.parent / f"feature_labels_sample{args.sample_random}{suffix}.json"
    else:
        out_path = in_path.parent / f"feature_labels{suffix}.json"

    data = json.loads(in_path.read_text())
    features = data["features"]
    live_ids = data["live_feature_ids"]
    if args.sample_random and args.sample_random < len(live_ids):
        import random
        rng = random.Random(args.seed)
        live_ids = rng.sample(live_ids, args.sample_random)
        live_ids.sort()
    elif args.max_features:
        live_ids = live_ids[: args.max_features]

    existing = {}
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text()).get("labels", {})
        print(f"resuming: {len(existing)} features already labeled")

    labels = dict(existing)
    counts = {k: 0 for k in ("COHERENT", "THEMATIC", "GENERIC",
                              "POLYSEMANTIC", "UNCLEAR")}
    for k in existing.values():
        counts[k["judgment"]] = counts.get(k["judgment"], 0) + 1

    t0 = time.monotonic()
    for i, fid in enumerate(live_ids):
        fid_s = str(fid)
        if fid_s in labels:
            continue
        hits = features[fid_s]
        try:
            r = label_one(fid_s, hits, args.model, args.max_windows,
                          prompt_variant=args.prompt_variant)
        except Exception as exc:
            print(f"  feature {fid} ERR: {exc}")
            continue
        labels[fid_s] = r
        counts[r["judgment"]] = counts.get(r["judgment"], 0) + 1

        # Persist every 10 features for crash-resilience
        if (i + 1) % 10 == 0 or i == len(live_ids) - 1:
            payload = {
                "source": str(in_path),
                "run": data["run"],
                "sae_type": data["sae_type"],
                "num_latents": data["num_latents"],
                "n_live_features": data["n_live_features"],
                "model": args.model,
                "max_windows": args.max_windows,
                "prompt_variant": args.prompt_variant,
                "counts": counts,
                "labels": labels,
            }
            out_path.write_text(json.dumps(payload, indent=2))

        elapsed = time.monotonic() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(live_ids) - i - 1) / rate if rate > 0 else float("inf")
        print(f"  [{i+1:3d}/{len(live_ids)}] f{fid} → {r['judgment']:<13} "
              f"({rate:.1f}/s, eta {eta/60:.1f}m)")

    print(f"\n=== {data['run']} — feature quality breakdown ===")
    total = sum(counts.values())
    for cat in ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"):
        c = counts.get(cat, 0)
        print(f"  {cat:<13} {c:>4}  ({100*c/max(total,1):.1f}%)")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
