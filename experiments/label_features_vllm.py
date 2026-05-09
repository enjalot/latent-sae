"""Label SAE features via a concurrent OpenAI-compatible API.

Mirrors `label_features_ollama.py` exactly — same DESCRIBE / JUDGE prompts,
same five-bucket taxonomy — but sends requests to an OpenAI-compatible
/v1/completions endpoint with thread-pool concurrency so the server can
batch internally on the GPU.

Works against either:
  - ollama with OLLAMA_NUM_PARALLEL >= 4 (port 11434, model "qwen2.5:32b-instruct-q4_K_M")
  - vllm OpenAI server (port 8000, model "Qwen/Qwen2.5-32B-Instruct-AWQ")

Concurrency lets the backend batch multiple sequences in one forward pass.
Ollama with NUM_PARALLEL=4 typically gets ~3× the throughput of serial
single-request labeling on the same model.

Usage:
    python -m experiments.label_features_vllm \\
        --input experiments/results/<run>/feature_activations.json \\
        --model qwen2.5:32b-instruct-q4_K_M \\
        --api-url http://127.0.0.1:11434 \\
        --concurrency 4
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


DEFAULT_API_URL = "http://127.0.0.1:11434"


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


VALID_LABELS = {"COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"}


def vllm_complete(api_url: str, model: str, prompt: str, max_tokens: int = 200,
                  temperature: float = 0.2, timeout: int = 300) -> str:
    """Hit vllm's OpenAI-compatible /v1/completions endpoint."""
    r = requests.post(f"{api_url}/v1/completions", json={
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": False,
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


def label_one(feature_id: str, windows: list[dict], model: str, api_url: str,
              max_windows: int = 10, prompt_variant: str = "default",
              describe_only: bool = False) -> dict:
    sel = windows[:max_windows]
    formatted = "\n".join(
        f"{i+1}. [act={w['activation']:.2f}] {w['window']}"
        for i, w in enumerate(sel)
    )
    tmpl = POSITIONAL_DESCRIBE_TEMPLATE if prompt_variant == "positional" else DESCRIBE_TEMPLATE
    prompt = tmpl.format(n=len(sel), fid=feature_id, windows=formatted)
    desc = vllm_complete(api_url, model, prompt, max_tokens=200, temperature=0.2)

    if describe_only:
        # Skip the JUDGE call. Caller will assign judgment via cheap classifier.
        return {"description": desc, "judgment": "PENDING"}

    judge_prompt = JUDGE_TEMPLATE.format(description=desc)
    raw = vllm_complete(api_url, model, judge_prompt, max_tokens=10, temperature=0.0)
    label = raw.split()[0].strip().upper() if raw else "UNCLEAR"
    if label not in VALID_LABELS:
        # Try to find a valid token in the response
        for tok in raw.upper().split():
            if tok.strip(".,!?:;") in VALID_LABELS:
                label = tok.strip(".,!?:;")
                break
        else:
            label = "UNCLEAR"
    return {"description": desc, "judgment": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to feature_activations.json")
    ap.add_argument("--model", default="qwen2.5:32b-instruct-q4_K_M",
                    help="model name as known to the backend (ollama tag or HF id)")
    ap.add_argument("--api-url", default=DEFAULT_API_URL,
                    help="OpenAI-compatible /v1 base URL (no trailing slash)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel in-flight feature jobs (backend batches them)")
    ap.add_argument("--max-windows", type=int, default=10)
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--sample-random", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prompt-variant", default="default",
                    choices=["default", "positional"])
    ap.add_argument("--persist-every", type=int, default=200,
                    help="write JSON every N completed features (default 200)")
    ap.add_argument("--describe-only", action="store_true",
                    help="Skip the 32b JUDGE call. Each feature's judgment is "
                         "left as PENDING. Use cheap_classifier_v3 + apply_cheap_classifier "
                         "afterwards to assign final labels. Halves LLM cost per feature.")
    args = ap.parse_args()

    in_path = Path(args.input)
    suffix = "_positional" if args.prompt_variant == "positional" else ""
    suffix += "_concurrent"
    if args.describe_only:
        suffix += "_describeonly"
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

    todo_ids = [fid for fid in live_ids if str(fid) not in existing]
    print(f"to label: {len(todo_ids)} features (skipping {len(live_ids) - len(todo_ids)} already done)")

    labels = dict(existing)
    counts = {k: 0 for k in ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR", "PENDING")}
    for k in existing.values():
        counts[k["judgment"]] = counts.get(k["judgment"], 0) + 1

    t0 = time.monotonic()
    completed = 0
    last_persist = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {}
        for fid in todo_ids:
            fid_s = str(fid)
            hits = features[fid_s]
            fut = pool.submit(label_one, fid_s, hits, args.model, args.api_url,
                              args.max_windows, args.prompt_variant,
                              args.describe_only)
            futs[fut] = fid_s

        for fut in as_completed(futs):
            fid_s = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                print(f"  feature {fid_s} ERR: {exc}")
                continue
            labels[fid_s] = rec
            counts[rec["judgment"]] = counts.get(rec["judgment"], 0) + 1
            completed += 1

            if completed - last_persist >= args.persist_every:
                last_persist = completed
                _persist(out_path, in_path, data, args, counts, labels)

            elapsed = time.monotonic() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(todo_ids) - completed) / rate if rate > 0 else float("inf")
            if completed % 50 == 0 or completed == len(todo_ids):
                print(f"  [{completed:>5}/{len(todo_ids)}] f{fid_s} → {rec['judgment']:<13} "
                      f"rate={rate:.2f}/s eta={eta/60:.1f}m")

    # Final write
    _persist(out_path, in_path, data, args, counts, labels)

    elapsed = time.monotonic() - t0
    rate = completed / elapsed if elapsed > 0 else 0
    print(f"\n=== {data['run']} — feature quality breakdown ===")
    total = sum(counts.values())
    for cat in ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR", "PENDING"):
        c = counts.get(cat, 0)
        if c > 0 or cat != "PENDING":
            print(f"  {cat:<13} {c:>5}  ({100*c/max(total,1):.1f}%)")
    print(f"\nthroughput: {rate:.2f} features/sec ({elapsed:.0f}s for {completed} features)")
    print(f"  serial baseline ≈ 0.69 features/sec (1.45 s/feature)")
    print(f"  speedup ≈ {rate / 0.69:.2f}× over serial")
    print(f"wrote {out_path}")


def _persist(out_path, in_path, data, args, counts, labels):
    payload = {
        "source": str(in_path),
        "run": data["run"],
        "sae_type": data["sae_type"],
        "num_latents": data["num_latents"],
        "n_live_features": data["n_live_features"],
        "model": args.model,
        "backend": "concurrent_openai",
        "api_url": args.api_url,
        "concurrency": args.concurrency,
        "max_windows": args.max_windows,
        "prompt_variant": args.prompt_variant,
        "counts": counts,
        "labels": labels,
    }
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
