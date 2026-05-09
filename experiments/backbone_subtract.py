"""Backbone-subtracted MaxSim attribution.

Many features in 17r fire on essentially any academic prose ("variety
of content words", "academic terminology", etc). They dominate the
per-query attribution rankings even though they aren't query-distinctive.
This script computes the average per-feature attribution across a SET of
queries (the "backbone") and re-ranks each query's features by their
DELTA from that average — surfacing features that are unusually active
for that specific query.

Reads the attribution JSON produced by attribute_query_set.py and
writes a backbone-subtracted re-ranking.

Usage:
    python -m experiments.backbone_subtract \\
        --attribution /data/embeddings/beir/trec-covid-attribution.json \\
        --top-n 10
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True,
                    help="JSON written by attribute_query_set.py")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--metric", default="max_attribution",
                    choices=["max_attribution", "sum_attribution", "binary_score"],
                    help="which raw attribution metric to subtract from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.attribution).read_text())
    records = data["records"]
    n_records = len(records)
    if n_records < 2:
        raise SystemExit("need ≥2 records to compute backbone")

    # 1. Compute per-feature mean attribution across all queries.
    # Note: a feature may not appear in every query's top-200; treat
    # missing as 0 for that query.
    per_feature_per_query: dict[int, dict[str, float]] = defaultdict(dict)
    all_features = set()
    for rec in records:
        qid = rec["qid"]
        for fr in rec.get("top_all", []):
            f = fr["feature"]
            per_feature_per_query[f][qid] = float(fr[args.metric])
            all_features.add(f)

    # Mean across queries (treating missing as 0)
    backbone_mean: dict[int, float] = {}
    for f in all_features:
        vals = [per_feature_per_query[f].get(rec["qid"], 0.0) for rec in records]
        backbone_mean[f] = float(np.mean(vals))

    # 2. For each query, compute delta = own - backbone_mean and re-rank.
    # Also load labels from sae_dir to enrich descriptions
    sae_dir = Path(data["sae_dir"])
    labels = {}
    p = sae_dir / "feature_labels.json"
    if p.exists():
        labels = json.loads(p.read_text())["labels"]

    out_records = []
    for rec in records:
        qid = rec["qid"]
        own = {fr["feature"]: float(fr[args.metric]) for fr in rec.get("top_all", [])}
        # Compute delta for any feature that fired in this query
        deltas = []
        for f, v in own.items():
            d = v - backbone_mean.get(f, 0.0)
            deltas.append({"feature": f, "delta": d, "own": v,
                           "backbone": backbone_mean.get(f, 0.0)})
        deltas.sort(key=lambda x: -x["delta"])

        top = []
        for entry in deltas[: args.top_n]:
            f = entry["feature"]
            lab = labels.get(str(f), {})
            top.append({**entry, "judgment": lab.get("judgment", ""),
                         "description": lab.get("description", "")})
        out_records.append({
            "qid": qid, "did": rec["did"],
            "query_text": rec.get("query_text", ""),
            "doc_title": rec.get("doc_title_text", "")[:120],
            "score": rec["score"],
            "ranks": rec.get("ranks", {}),
            "backbone_subtracted_top": top,
        })

        # Print
        print("\n" + "=" * 90)
        print(f"qid={qid}  did={rec['did']}  MaxSim={rec['score']:.2f}")
        print(f"  query: {rec.get('query_text','')[:140]}")
        print(f"  doc:   {rec.get('doc_title_text','')[:140]}")
        print(f"\n  Top {len(top)} backbone-subtracted features (delta={args.metric}):")
        print(f"  {'rank':>4} {'fid':>7} {'delta':>7} {'own':>6} {'bbone':>6}  description")
        for i, e in enumerate(top):
            d = e["delta"]; o = e["own"]; b = e["backbone"]
            j = e["judgment"][:4]
            desc = (e["description"] or "(no label)")[:80]
            print(f"  {i+1:>4} {e['feature']:>7} {d:>+7.3f} {o:>6.2f} {b:>6.2f}  [{j}] {desc}")

    out_path = (Path(args.out) if args.out
                else Path(args.attribution).with_name(
                    Path(args.attribution).stem + "-backbone-subtracted.json"))
    out_path.write_text(json.dumps({
        "source": str(args.attribution),
        "metric": args.metric,
        "n_queries": n_records,
        "n_unique_features": len(all_features),
        "records": out_records,
    }, indent=2))
    print(f"\nwrote {out_path}")
    # Also print the top "always-on backbone" features for context
    print("\n=== Top 10 backbone features (always-on across queries) ===")
    bb = sorted(backbone_mean.items(), key=lambda x: -x[1])[:10]
    for f, m in bb:
        lab = labels.get(str(f), {})
        j = lab.get("judgment", "")[:4]
        desc = (lab.get("description", "") or "")[:80]
        print(f"  f{f:<6} mean_{args.metric[:3]}={m:>5.2f}  [{j}] {desc}")


if __name__ == "__main__":
    main()
