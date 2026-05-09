"""Apply a saved cheap classifier (v3) to a labels file with PENDING judgments.

Inputs:
  - --run-dir: SAE run directory (must contain feature_activations.json,
               feature_labels_concurrent_describeonly.json,
               feature_topic_cohesion_scores.json,
               feature_embedsim_scores.json)
  - --classifier: path to a joblib file produced by cheap_classifier_v3
                  containing keys "logreg", "tfidf", "numeric_names"

Output:
  - Writes feature_labels_cheap.json (same shape as feature_labels.json) with
    judgments produced by the classifier.
  - Symlinks feature_labels.json -> feature_labels_cheap.json so downstream
    scripts (extract / rigor / etc) Just Work.
  - Prints distribution + binary "interpretable" rate.

Usage:
    python -m experiments.apply_cheap_classifier \\
        --run-dir experiments/results/<run> \\
        --classifier experiments/classifiers/cheap_v3_17r.joblib
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse featurize logic
from cheap_classifier_v3 import numeric_features, assemble_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--classifier", required=True,
                    help="path to cheap_v3 .joblib")
    ap.add_argument("--describeonly-labels",
                    default="feature_labels_concurrent_describeonly.json",
                    help="filename of describe-only labels file in run dir")
    ap.add_argument("--out", default="feature_labels_cheap.json",
                    help="output filename (in run dir)")
    ap.add_argument("--symlink", action="store_true", default=True,
                    help="also symlink feature_labels.json to the output")
    args = ap.parse_args()

    rd = Path(args.run_dir)
    print(f"loading classifier from {args.classifier}")
    import joblib
    bundle = joblib.load(args.classifier)
    clf = bundle["logreg"]
    tfidf = bundle["tfidf"]

    # Construct records by hand from describe-only labels file (since
    # assemble_records would normally read feature_labels.json)
    desc_labels = json.loads((rd / args.describeonly_labels).read_text())["labels"]
    coh_per = {}
    coh_path = rd / "feature_topic_cohesion_scores.json"
    if coh_path.exists():
        for r in json.loads(coh_path.read_text()).get("per_feature", []):
            coh_per[r["feature"]] = r
    es_per = {}
    es_path = rd / "feature_embedsim_scores.json"
    if es_path.exists():
        for r in json.loads(es_path.read_text()).get("per_feature", []):
            es_per[r["feature"]] = r
    acts = json.loads((rd / "feature_activations.json").read_text())
    n_hits = {fid: len(hits) for fid, hits in acts["features"].items()}

    records = []
    for fid, rec in desc_labels.items():
        c = coh_per.get(fid, {})
        e = es_per.get(fid, {})
        records.append({
            "feature": fid,
            "judgment": rec["judgment"],
            "description": rec["description"],
            "cohesion_gap": c.get("gap", 0.0),
            "cohesion_auc": c.get("auc", 0.5),
            "embedsim_auc": e.get("auc", 0.5),
            "embedsim_mean_pos": e.get("mean_pos_sim", 0.0),
            "embedsim_mean_neg": e.get("mean_neg_sim", 0.0),
            "n_hits": n_hits.get(fid, 0),
        })
    print(f"{len(records)} features to classify")

    X_num, num_names = numeric_features(records)
    descs = [r["description"] for r in records]
    X_text = tfidf.transform(descs)

    from scipy.sparse import hstack, csr_matrix
    X = hstack([csr_matrix(X_num), X_text])

    print("predicting ...")
    y_pred = clf.predict(X)

    out_labels = {}
    counts = {k: 0 for k in ("COHERENT", "THEMATIC", "GENERIC",
                              "POLYSEMANTIC", "UNCLEAR")}
    for rec, pred in zip(records, y_pred):
        out_labels[rec["feature"]] = {
            "description": rec["description"],
            "judgment": str(pred),
            "judgment_source": "cheap_classifier_v3",
        }
        counts[str(pred)] = counts.get(str(pred), 0) + 1

    payload = {
        "source_describe_only": str(rd / args.describeonly_labels),
        "classifier": str(args.classifier),
        "run": acts["run"],
        "method": "cheap_classifier_v3",
        "counts": counts,
        "labels": out_labels,
    }
    out_path = rd / args.out
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"\n=== {acts['run']} cheap-classifier breakdown ===")
    total = sum(counts.values())
    for cat in ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"):
        c = counts.get(cat, 0)
        print(f"  {cat:<13} {c:>6}  ({100*c/max(total,1):.1f}%)")
    interp = counts.get("COHERENT", 0) + counts.get("THEMATIC", 0)
    print(f"  ---")
    print(f"  interpretable (COH+THM) {interp:>6}  ({100*interp/max(total,1):.1f}%)")

    if args.symlink:
        sym = rd / "feature_labels.json"
        if sym.exists() or sym.is_symlink():
            sym.unlink()
        sym.symlink_to(args.out)
        print(f"\nsymlinked {sym} -> {args.out}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
