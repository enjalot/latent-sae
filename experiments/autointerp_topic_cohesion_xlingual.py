"""Per-language topic cohesion scorer.

Adapts `autointerp_topic_cohesion.py` for multilingual SAEs:

- Reads `feature_activations_xlingual.json` (corpus-tagged top hits).
- Encodes windows with `paraphrase-multilingual-MiniLM-L12-v2` (50+ langs).
- For each feature, picks its **dominant corpus** (most-represented in
  top-N hits) and computes baseline against a per-corpus pool — so
  language_bound features are measured against same-language clutter
  rather than English-MiniLM-on-non-English nonsense.

Outputs:
  - per-feature gap / AUC (same shape as the English scorer)
  - per-corpus aggregate (mean gap, frac > 0.05, n features)
  - per-bucket aggregate (cross_lingual / language_bound / english_only)

Usage:
    python -m experiments.autointerp_topic_cohesion_xlingual \\
        --run-dir experiments/results/<run> \\
        [--n-features 0]   # 0 = score all eligible
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SENTENCE_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
SENTENCE_CACHE = "/data/hf/sentence-transformers"
TOKEN_MARKER_RE = re.compile(r"<<|>>")

EN_CORPORA = {"fineweb", "redpajama", "pile"}


def strip_markers(window: str) -> str:
    return TOKEN_MARKER_RE.sub("", window).strip()


def bucketize(shares: dict, threshold: float = 0.80) -> str:
    en_share = sum(v for k, v in shares.items() if k in EN_CORPORA)
    if en_share >= threshold:
        return "english_only"
    non_en = {k: v for k, v in shares.items() if k not in EN_CORPORA}
    if non_en and max(non_en.values()) >= threshold:
        return "language_bound"
    return "cross_lingual"


def encode_sentences(texts, model, batch_size=128):
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-features", type=int, default=0)
    ap.add_argument("--min-hits", type=int, default=10)
    ap.add_argument("--max-hits", type=int, default=16)
    ap.add_argument("--per-corpus-pool", type=int, default=200,
                    help="size of per-corpus baseline pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    xl_path = run_dir / "feature_activations_xlingual.json"
    if not xl_path.exists():
        raise FileNotFoundError(f"need xlingual activations at {xl_path}")
    data = json.loads(xl_path.read_text())
    features = data["features"]

    # Eligible features: enough hits
    eligible = [fid for fid, hits in features.items() if len(hits) >= args.min_hits]
    rng = random.Random(args.seed)
    if args.n_features > 0 and len(eligible) > args.n_features:
        rng.shuffle(eligible)
        eligible = eligible[:args.n_features]
    eligible.sort(key=int)
    print(f"scoring {len(eligible)} features")

    # Determine each feature's dominant corpus + bucket
    feat_dominant = {}
    feat_bucket = {}
    feat_shares = {}
    for fid in eligible:
        hits = features[fid][:args.max_hits]
        c = Counter(h["corpus"] for h in hits)
        n = sum(c.values())
        shares = {k: v / n for k, v in c.items()}
        feat_shares[fid] = shares
        feat_dominant[fid] = max(shares, key=shares.get)
        feat_bucket[fid] = bucketize(shares)

    # Build per-corpus baseline pools by sampling one window per
    # feature for any feature that has ≥1 hit in that corpus.
    # Each pool is sampled independently with seed.
    rng_pool = random.Random(args.seed + 1)
    per_corpus_features = defaultdict(list)
    for fid in eligible:
        for h in features[fid][:args.max_hits]:
            per_corpus_features[h["corpus"]].append((fid, h))

    pool_texts_by_corpus = {}
    for corpus, fid_hits in per_corpus_features.items():
        sample = rng_pool.sample(fid_hits,
                                 k=min(args.per_corpus_pool, len(fid_hits)))
        pool_texts_by_corpus[corpus] = [strip_markers(h.get("window", ""))
                                        for _, h in sample if h.get("window")]

    # Flatten all unique encode targets:
    #   - per-feature top-N windows
    #   - per-corpus baseline pool windows
    flat_texts = []
    feat_window_slices = {}
    pool_slices = {}

    for fid in eligible:
        windows = [strip_markers(h.get("window", ""))
                   for h in features[fid][:args.max_hits]]
        windows = [w for w in windows if w]
        s = len(flat_texts)
        flat_texts.extend(windows)
        e = len(flat_texts)
        feat_window_slices[fid] = (s, e)

    for corpus, texts in pool_texts_by_corpus.items():
        s = len(flat_texts)
        flat_texts.extend(texts)
        e = len(flat_texts)
        pool_slices[corpus] = (s, e)

    print(f"encoding {len(flat_texts)} total windows ({len(eligible)} features + "
          f"{sum(len(v) for v in pool_texts_by_corpus.values())} pool entries "
          f"across {len(pool_texts_by_corpus)} corpora)")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SENTENCE_MODEL_ID, cache_folder=SENTENCE_CACHE,
                                device=args.device)
    t0 = time.monotonic()
    vecs = encode_sentences(flat_texts, model)
    print(f"encoded in {time.monotonic() - t0:.1f}s; shape={vecs.shape}")

    pool_vecs_by_corpus = {
        c: vecs[s:e] for c, (s, e) in pool_slices.items() if e > s
    }

    per_feature = []
    for fid in eligible:
        s, e = feat_window_slices[fid]
        if e - s < 3:
            continue
        own = vecs[s:e]

        sim = own @ own.T
        n = own.shape[0]
        mask = ~np.eye(n, dtype=bool)
        within = float(sim[mask].mean())

        dom = feat_dominant[fid]
        pool = pool_vecs_by_corpus.get(dom)
        if pool is None or pool.shape[0] < 5:
            # fall back to fineweb pool (always populated)
            pool = pool_vecs_by_corpus.get("fineweb")
        if pool is None:
            continue

        cross = own @ pool.T
        baseline = float(cross.mean())
        gap = within - baseline

        within_pairs = sim[mask]
        cross_pairs = cross.flatten()
        all_vals = np.concatenate([within_pairs, cross_pairs])
        ranks = np.argsort(np.argsort(all_vals)) + 1
        n_w = len(within_pairs); n_c = len(cross_pairs)
        u = ranks[:n_w].sum() - n_w * (n_w + 1) / 2
        auc = float(u / (n_w * n_c))

        rec = {
            "feature": fid, "n_windows": n,
            "dominant_corpus": dom,
            "bucket": feat_bucket[fid],
            "within_cos": within,
            "baseline_cos": baseline,
            "gap": gap,
            "auc": auc,
        }
        per_feature.append(rec)

    # Aggregate
    by_corpus = defaultdict(list)
    by_bucket = defaultdict(list)
    for r in per_feature:
        by_corpus[r["dominant_corpus"]].append(r)
        by_bucket[r["bucket"]].append(r)

    def agg(rs):
        gaps = [r["gap"] for r in rs]
        aucs = [r["auc"] for r in rs]
        return {
            "n": len(rs),
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "frac_gap_above_0.05": float(np.mean([g > 0.05 for g in gaps])),
            "mean_auc": float(np.mean(aucs)),
        }

    overall_within = float(np.mean([r["within_cos"] for r in per_feature]))
    overall_baseline = float(np.mean([r["baseline_cos"] for r in per_feature]))
    overall_gap = overall_within - overall_baseline
    overall_auc = float(np.mean([r["auc"] for r in per_feature]))

    report = {
        "run": run_dir.name,
        "scorer": "topic_cohesion_xlingual",
        "encoder": SENTENCE_MODEL_ID,
        "n_features_scored": len(per_feature),
        "overall_within_cos": overall_within,
        "overall_baseline_cos": overall_baseline,
        "overall_gap": overall_gap,
        "overall_auc": overall_auc,
        "per_corpus": {c: agg(rs) for c, rs in by_corpus.items()},
        "per_bucket": {b: agg(rs) for b, rs in by_bucket.items()},
        "per_feature": per_feature,
    }
    out = run_dir / "feature_topic_cohesion_xlingual_scores.json"
    out.write_text(json.dumps(report, indent=2))

    print()
    print(f"=== {run_dir.name} ===")
    print(f"overall: within={overall_within:.3f} baseline={overall_baseline:.3f} "
          f"gap={overall_gap:+.3f} AUC={overall_auc:.3f}")
    print(f"\n{'bucket':<16} {'n':>6} {'gap':>8} {'>0.05':>8} {'AUC':>7}")
    for b in ("english_only", "language_bound", "cross_lingual"):
        if b not in by_bucket:
            continue
        a = agg(by_bucket[b])
        print(f"{b:<16} {a['n']:>6} {a['mean_gap']:>+8.3f} "
              f"{a['frac_gap_above_0.05']:>7.1%} {a['mean_auc']:>7.3f}")

    print(f"\n{'corpus':<24} {'n':>6} {'gap':>8} {'>0.05':>8} {'AUC':>7}")
    for c in sorted(by_corpus, key=lambda x: -len(by_corpus[x])):
        a = agg(by_corpus[c])
        if a["n"] < 50:
            break
        print(f"{c:<24} {a['n']:>6} {a['mean_gap']:>+8.3f} "
              f"{a['frac_gap_above_0.05']:>7.1%} {a['mean_auc']:>7.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
