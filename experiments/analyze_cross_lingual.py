"""Analyze per-feature corpus distribution from a feature_activations_xlingual.json.

For each live feature, count how many of its top-K hits come from each
corpus, derive normalized share + entropy, and bucket as:

  - english_only      — >=80% top hits in EN (fineweb/redpajama/pile)
  - language_bound    — >=80% top hits in a single non-EN ml_<lang>
  - cross_lingual     — neither (multi-language spread)

Output:
  - feature_xlingual_summary.json — per-feature share, entropy, bucket
  - prints histogram + means per bucket

Usage:
    python -m experiments.analyze_cross_lingual \\
        --xlingual experiments/results/<run>/feature_activations_xlingual.json \\
        --cohesion experiments/results/<run>/feature_topic_cohesion_scores.json
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


EN_CORPORA = {"fineweb", "redpajama", "pile"}


def bucketize(shares: dict, threshold: float = 0.80):
    en_share = sum(v for k, v in shares.items() if k in EN_CORPORA)
    if en_share >= threshold:
        return "english_only"
    non_en = {k: v for k, v in shares.items() if k not in EN_CORPORA}
    if non_en and max(non_en.values()) >= threshold:
        return "language_bound"
    return "cross_lingual"


def shannon_entropy(shares: dict) -> float:
    return -sum(p * math.log(p, 2) for p in shares.values() if p > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlingual", required=True)
    ap.add_argument("--cohesion", default=None,
                    help="optional cohesion JSON for per-bucket cohesion stats")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    xl_path = Path(args.xlingual)
    data = json.loads(xl_path.read_text())
    features = data["features"]

    cohesion_by_fid = {}
    if args.cohesion and Path(args.cohesion).exists():
        coh = json.loads(Path(args.cohesion).read_text())
        for r in coh.get("per_feature", []):
            fid = r.get("feature_id") or r.get("fid") or r.get("feature")
            if fid is None:
                continue
            cohesion_by_fid[str(fid)] = r.get("gap", None)

    bucket_counts = Counter()
    bucket_cohesion: dict[str, list[float]] = {"english_only": [],
                                                "language_bound": [],
                                                "cross_lingual": []}
    bucket_corpora: dict[str, Counter] = {"english_only": Counter(),
                                          "language_bound": Counter(),
                                          "cross_lingual": Counter()}
    per_feature = {}

    for fid, hits in features.items():
        if not hits:
            continue
        c = Counter(h["corpus"] for h in hits)
        n = sum(c.values())
        shares = {k: v / n for k, v in c.items()}
        bucket = bucketize(shares)
        ent = shannon_entropy(shares)
        per_feature[fid] = {
            "n_hits": n,
            "shares": shares,
            "dominant": max(shares, key=shares.get),
            "dominant_share": max(shares.values()),
            "entropy_bits": round(ent, 3),
            "n_corpora": len(shares),
            "bucket": bucket,
        }
        bucket_counts[bucket] += 1
        if fid in cohesion_by_fid and cohesion_by_fid[fid] is not None:
            bucket_cohesion[bucket].append(cohesion_by_fid[fid])
        bucket_corpora[bucket].update(c)

    total = sum(bucket_counts.values())
    print(f"\n=== Cross-lingual feature analysis ({total} live features) ===")
    print(f"  english_only:    {bucket_counts['english_only']:>6,}  "
          f"({bucket_counts['english_only']/total:.1%})")
    print(f"  language_bound:  {bucket_counts['language_bound']:>6,}  "
          f"({bucket_counts['language_bound']/total:.1%})")
    print(f"  cross_lingual:   {bucket_counts['cross_lingual']:>6,}  "
          f"({bucket_counts['cross_lingual']/total:.1%})")

    if cohesion_by_fid:
        print(f"\n=== Mean cohesion gap per bucket ===")
        for b, vals in bucket_cohesion.items():
            if vals:
                print(f"  {b:<16} n={len(vals):>5,}  "
                      f"mean_gap={sum(vals)/len(vals):.3f}  "
                      f"frac_gap>0.05={sum(1 for v in vals if v > 0.05)/len(vals):.1%}")

    print(f"\n=== Top 5 dominant corpora per bucket ===")
    for b in ("english_only", "language_bound", "cross_lingual"):
        top5 = bucket_corpora[b].most_common(5)
        print(f"  {b:<16} {top5}")

    print(f"\n=== Language-bound: count per language ===")
    lang_bound_dom = Counter(per_feature[fid]["dominant"]
                             for fid, info in per_feature.items()
                             if info["bucket"] == "language_bound")
    for slug, n in lang_bound_dom.most_common():
        print(f"  {slug:>22}: {n:>5,}")

    out_path = Path(args.out) if args.out else xl_path.parent / "feature_xlingual_summary.json"
    summary = {
        "run": data.get("run"),
        "n_features_analyzed": total,
        "buckets": dict(bucket_counts),
        "bucket_mean_cohesion_gap": {
            b: (sum(v) / len(v) if v else None)
            for b, v in bucket_cohesion.items()
        },
        "language_bound_per_language": dict(lang_bound_dom),
        "per_feature": per_feature,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
