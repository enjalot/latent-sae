"""Cheap rule-based classifier for SAE feature descriptions.

Inputs per feature:
  - description (str): the 32b-written description
  - topic_cohesion_gap (float): within-feature mean cosine - cross-feature baseline
  - topic_cohesion_auc (float): AUC of within-feature pairs vs cross-feature pairs
  - embedsim_auc (float, optional): AUC of cosine(desc, pos_window) vs cosine(desc, neg_window)
  - n_hits (int): how many activating windows the feature has (proxy for feature strength)

Output: one of COHERENT / THEMATIC / GENERIC / POLYSEMANTIC / UNCLEAR.

The intent is to replace the 32b JUDGE step (the second LLM call per feature)
with rules + cheap label-free scorers. If agreement with 32b is high enough
(Cohen's kappa >= 0.6 or so), we save half the labeling LLM time on 17b/17s.

Usage:
    python -m experiments.cheap_classifier classify-run --run-dir <run> [--out <path>]
    python -m experiments.cheap_classifier validate --gold <full_labels.json> [--scorers <run-dir>]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional


# -------- Rule patterns --------

POLY_PATTERNS = [
    r"\b(unrelated|polysemantic|disparate|disconnected)\b",
    r"\b(no\s+clear\s+(commonality|connection|pattern|shared|theme|concept))\b",
    r"\b(spans?\s+multiple\s+unrelated)\b",
    r"\b(diverse\s+(?:concepts|topics|patterns)\s+(?:without|with\s+no))\b",
    r"\b(does\s+not\s+seem\s+to\s+respond\s+consistently)\b",
    r"\b(activate(s|d)?\s+across\s+various\s+unrelated)\b",
]

GENERIC_PATTERNS = [
    r"\b(punctuation|stopwords?|stop[- ]words?|function\s+words?)\b",
    r"\b(sentence\s+(?:position|boundary|boundaries|breaks?))\b",
    r"\b(structural\s+markers?)\b",
    r"\b(generic\s+(markers?|tokens?))\b",
    r"\b(filler\s+words?|delim(it)?ers?)\b",
    r"\b(line\s+breaks?|paragraph\s+breaks?)\b",
]

UNCLEAR_PATTERNS = [
    r"\b(unclear|too\s+vague|cannot\s+determine|hard\s+to\s+identify)\b",
    r"\b(insufficient\s+(?:information|context))\b",
    r"\b(could\s+not\s+identify\s+a\s+(?:clear|specific))\b",
]


def first_match(patterns, text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def cheap_classify(description: str,
                   cohesion_gap: float,
                   cohesion_auc: float,
                   embedsim_auc: Optional[float] = None,
                   n_hits: int = 0) -> str:
    """Rule-based bucket assignment. Order matters: stronger evidence first."""
    desc = description or ""
    desc_low = desc.lower()

    # Hard text signals
    if first_match(POLY_PATTERNS, desc):
        return "POLYSEMANTIC"
    if first_match(UNCLEAR_PATTERNS, desc):
        return "UNCLEAR"
    if first_match(GENERIC_PATTERNS, desc):
        return "GENERIC"

    # Soft scorer signals
    # Strong cohesion + strong description-context match → COHERENT
    if cohesion_auc >= 0.58 and (embedsim_auc is None or embedsim_auc >= 0.70):
        if cohesion_gap >= 0.04:
            return "COHERENT"
    # Moderate cohesion → THEMATIC
    if cohesion_auc >= 0.55 and cohesion_gap >= 0.02:
        return "THEMATIC"

    # Weak cohesion + few hits → GENERIC fallback (likely a degenerate feature)
    if cohesion_gap < 0.02 and n_hits >= 5:
        return "GENERIC"

    # Default for ambiguous middle
    return "POLYSEMANTIC"


# -------- Driver: classify a run's features given existing scorer outputs --------

def classify_run(run_dir: Path, out_path: Optional[Path] = None) -> dict:
    """Apply cheap classifier to every labeled feature in run_dir.
    Reads feature_labels(_concurrent).json, feature_topic_cohesion_scores.json,
    feature_embedsim_scores.json, feature_activations.json (for n_hits).
    Writes a feature_labels_cheap.json with same shape as feature_labels.json
    but with cheap-classifier judgments.
    """
    # Find the description-bearing label file
    candidates = [run_dir / n for n in
                  ["feature_labels_concurrent.json", "feature_labels.json",
                   "feature_labels_sample512.json"]]
    labels_path = next((p for p in candidates if p.exists()), None)
    if not labels_path:
        raise FileNotFoundError(f"no labels file in {run_dir}")
    print(f"using descriptions from: {labels_path.name}")
    descs = json.loads(labels_path.read_text())["labels"]

    # Topic cohesion: keyed by feature, has within_cos / baseline_cos / gap / auc
    coh_path = run_dir / "feature_topic_cohesion_scores.json"
    coh_per = {}
    if coh_path.exists():
        coh = json.loads(coh_path.read_text())
        for r in coh.get("per_feature", []):
            coh_per[r["feature"]] = r

    # Embedsim: per_feature has feature, label, mean_pos_sim, mean_neg_sim, auc
    es_path = run_dir / "feature_embedsim_scores.json"
    es_per = {}
    if es_path.exists():
        es = json.loads(es_path.read_text())
        for r in es.get("per_feature", []):
            es_per[r["feature"]] = r

    # Activations: count of hits per feature
    acts = json.loads((run_dir / "feature_activations.json").read_text())
    n_hits = {fid: len(hits) for fid, hits in acts["features"].items()}

    out_labels = {}
    counts = {k: 0 for k in ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR")}
    for fid, rec in descs.items():
        desc = rec["description"]
        c = coh_per.get(fid, {})
        e = es_per.get(fid, {})
        bucket = cheap_classify(
            desc,
            cohesion_gap=c.get("gap", 0.0),
            cohesion_auc=c.get("auc", 0.5),
            embedsim_auc=e.get("auc"),
            n_hits=n_hits.get(fid, 0),
        )
        out_labels[fid] = {"description": desc, "judgment": bucket,
                           "judgment_source": "cheap_rule"}
        counts[bucket] = counts.get(bucket, 0) + 1

    payload = {
        "source": str(labels_path),
        "run": acts["run"],
        "method": "cheap_rule_classifier_v1",
        "counts": counts,
        "labels": out_labels,
    }
    if out_path is None:
        out_path = run_dir / "feature_labels_cheap.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


# -------- Validation: cheap vs gold (full 32b labels) --------

def cohens_kappa(items_a, items_b):
    cats = sorted(set(items_a) | set(items_b))
    n = len(items_a)
    obs = sum(1 for a, b in zip(items_a, items_b) if a == b) / n
    pa = Counter(items_a); pb = Counter(items_b)
    exp = sum((pa[c]/n) * (pb[c]/n) for c in cats)
    if exp >= 1: return 1.0
    return (obs - exp) / (1 - exp)


def validate(gold_path: Path, cheap_path: Path):
    gold = json.loads(gold_path.read_text())
    cheap = json.loads(cheap_path.read_text())
    g = gold["labels"]
    c = cheap["labels"]
    common = sorted(set(g) & set(c), key=lambda x: int(x))
    if not common:
        print("ERROR: no overlapping features")
        return
    g_lab = [g[fid]["judgment"] for fid in common]
    c_lab = [c[fid]["judgment"] for fid in common]
    print(f"\nCompared on {len(common)} features.")

    print("\n=== Distribution ===")
    buckets = ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR")
    print(f"{'bucket':<13} {'gold(32b)':>10} {'cheap':>8}")
    for b in buckets:
        gn = sum(1 for x in g_lab if x == b)
        cn = sum(1 for x in c_lab if x == b)
        print(f"{b:<13} {gn:>10} {cn:>8}")

    raw = sum(1 for a, b in zip(g_lab, c_lab) if a == b) / len(common)
    kappa = cohens_kappa(g_lab, c_lab)
    print(f"\nraw match rate:      {raw:.3f}  ({sum(1 for a,b in zip(g_lab, c_lab) if a==b)}/{len(common)})")
    print(f"Cohen's kappa:       {kappa:.3f}")
    print(f"  > 0.80 → strong; 0.60-0.80 substantial; < 0.60 questionable")

    # Per-class precision/recall
    print("\n=== Per-class precision/recall (treat gold as truth) ===")
    print(f"{'bucket':<13} {'precision':>10} {'recall':>8} {'f1':>6}")
    for b in buckets:
        tp = sum(1 for a, x in zip(g_lab, c_lab) if a == b and x == b)
        fp = sum(1 for a, x in zip(g_lab, c_lab) if a != b and x == b)
        fn = sum(1 for a, x in zip(g_lab, c_lab) if a == b and x != b)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2*p*r / max(p + r, 1e-9)
        print(f"{b:<13} {p:>10.3f} {r:>8.3f} {f1:>6.3f}")

    # Confusion matrix
    print("\n=== Confusion (rows=gold(32b), cols=cheap) ===")
    header = "".join(f"{b:>13}" for b in buckets)
    print(f"{'':<14}{header}")
    for gb in buckets:
        row = [str(sum(1 for a, x in zip(g_lab, c_lab) if a == gb and x == cb)) for cb in buckets]
        print(f"{gb:<14}" + "".join(f"{v:>13}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cls = sub.add_parser("classify-run", help="Apply cheap classifier to a run")
    cls.add_argument("--run-dir", required=True)
    cls.add_argument("--out", default=None)
    val = sub.add_parser("validate", help="Compare cheap classifications vs gold")
    val.add_argument("--gold", required=True, help="full 32b-judged labels JSON")
    val.add_argument("--cheap", required=True, help="cheap-classifier labels JSON")

    args = ap.parse_args()
    if args.cmd == "classify-run":
        rd = Path(args.run_dir)
        out = Path(args.out) if args.out else None
        rep = classify_run(rd, out)
        print(f"\nwrote {rd / 'feature_labels_cheap.json' if out is None else out}")
        print(f"counts: {rep['counts']}")
    else:
        validate(Path(args.gold), Path(args.cheap))


if __name__ == "__main__":
    main()
