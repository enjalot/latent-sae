"""Data-driven cheap classifier for SAE feature buckets.

Inputs per feature:
  - cohesion gap, AUC (label-free, MiniLM)
  - embedsim AUC + per-feature pos/neg sims (uses description)
  - description text features (length, key-token presence)
  - n_hits

Trains a small scikit classifier on a held-out split of the 32b judge labels
on 17r's 22,454 features. Reports kappa, raw match, per-class P/R, confusion.

If kappa >= 0.6, the cheap method is good enough to replace the 32b judge
step on 17b/17s.

Usage:
    python -m experiments.cheap_classifier_v2 --run-dir <17r dir> [--test-frac 0.3]
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# --- regex signals (kept from v1, but the model decides their weight) ---
POLY_RE = re.compile(
    r"\b(unrelated|polysemantic|disparate|disconnected|"
    r"no\s+clear\s+(?:commonality|connection|pattern|shared|theme|concept)|"
    r"spans?\s+multiple\s+unrelated|"
    r"diverse\s+(?:concepts|topics|patterns)\s+(?:without|with\s+no)|"
    r"does\s+not\s+seem\s+to\s+respond\s+consistently|"
    r"activate(?:s|d)?\s+across\s+various\s+unrelated)\b", re.I)
GEN_RE = re.compile(
    r"\b(punctuation|stopwords?|stop[- ]words?|function\s+words?|"
    r"sentence\s+(?:position|boundary|boundaries|breaks?)|"
    r"structural\s+markers?|generic\s+(?:markers?|tokens?)|"
    r"filler\s+words?|delim(?:it)?ers?|"
    r"line\s+breaks?|paragraph\s+breaks?)\b", re.I)
UNCL_RE = re.compile(
    r"\b(unclear|too\s+vague|cannot\s+determine|hard\s+to\s+identify|"
    r"insufficient\s+(?:information|context)|"
    r"could\s+not\s+identify\s+a\s+(?:clear|specific)|"
    r"too\s+vague\s+to\s+judge)\b", re.I)


def featurize(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build a feature matrix from per-feature records.

    Each record needs: description, cohesion_gap, cohesion_auc, embedsim_auc,
    embedsim_mean_pos, embedsim_mean_neg, n_hits.
    Returns (X, feature_names).
    """
    feature_names = [
        "cohesion_gap", "cohesion_auc",
        "embedsim_auc", "embedsim_mean_pos", "embedsim_mean_neg", "embedsim_pos_neg_gap",
        "n_hits", "n_hits_log",
        "desc_len", "desc_word_count", "desc_word_count_log",
        "has_poly_word", "has_gen_word", "has_uncl_word",
        "has_specific_concept",  # heuristic for COH-like
    ]
    rows = []
    for r in records:
        desc = r.get("description", "")
        words = desc.split()
        wlen = len(words)
        # "specific concept" heuristic: presence of multi-word noun-phrase-ish
        # signals that suggest a coherent topic
        has_specific = int(bool(re.search(
            r"\b(specifically|particular(?:ly)?|terms?\s+related\s+to|"
            r"refers?\s+to|primarily\s+respond)\b", desc, re.I)))

        rows.append([
            r.get("cohesion_gap", 0.0),
            r.get("cohesion_auc", 0.5),
            r.get("embedsim_auc", 0.5),
            r.get("embedsim_mean_pos", 0.0),
            r.get("embedsim_mean_neg", 0.0),
            r.get("embedsim_mean_pos", 0.0) - r.get("embedsim_mean_neg", 0.0),
            r.get("n_hits", 0),
            np.log1p(r.get("n_hits", 0)),
            len(desc),
            wlen,
            np.log1p(wlen),
            int(bool(POLY_RE.search(desc))),
            int(bool(GEN_RE.search(desc))),
            int(bool(UNCL_RE.search(desc))),
            has_specific,
        ])
    X = np.asarray(rows, dtype=np.float32)
    return X, feature_names


def assemble_records(run_dir: Path) -> list[dict]:
    """Pull together per-feature data from labels + scorers + activations."""
    labels_path = run_dir / "feature_labels.json"
    if not labels_path.exists() or labels_path.is_symlink():
        # follow symlink
        labels_path = labels_path.resolve()
    if not labels_path.exists():
        labels_path = run_dir / "feature_labels_concurrent.json"
    labels = json.loads(labels_path.read_text())["labels"]

    coh_per = {}
    coh_path = run_dir / "feature_topic_cohesion_scores.json"
    if coh_path.exists():
        for r in json.loads(coh_path.read_text()).get("per_feature", []):
            coh_per[r["feature"]] = r

    es_per = {}
    es_path = run_dir / "feature_embedsim_scores.json"
    if es_path.exists():
        for r in json.loads(es_path.read_text()).get("per_feature", []):
            es_per[r["feature"]] = r

    acts = json.loads((run_dir / "feature_activations.json").read_text())
    n_hits = {fid: len(hits) for fid, hits in acts["features"].items()}

    records = []
    for fid, rec in labels.items():
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
    return records


def cohens_kappa(items_a, items_b):
    cats = sorted(set(items_a) | set(items_b))
    n = len(items_a)
    obs = sum(1 for a, b in zip(items_a, items_b) if a == b) / n
    pa = Counter(items_a); pb = Counter(items_b)
    exp = sum((pa[c]/n) * (pb[c]/n) for c in cats)
    return 1.0 if exp >= 1 else (obs - exp) / (1 - exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="logreg",
                    choices=["logreg", "tree", "rf", "gbm"])
    ap.add_argument("--out-classifier", default=None,
                    help="path to write fitted model + feature spec (joblib)")
    args = ap.parse_args()

    rd = Path(args.run_dir)
    print(f"loading from: {rd}")
    records = assemble_records(rd)
    print(f"{len(records)} feature records assembled")

    # Filter records that have at least cohesion data (need ≥10 hits to be cohesion-eligible)
    has_signal = [r for r in records if r["n_hits"] >= 1]
    print(f"{len(has_signal)} with at least one hit")

    X, names = featurize(has_signal)
    y = np.array([r["judgment"] for r in has_signal])

    # Train/test split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(has_signal))
    rng.shuffle(idx)
    n_test = int(len(idx) * args.test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"train: {len(y_tr)}, test: {len(y_te)}")

    print(f"\nclass distribution (train):")
    for cls, cnt in Counter(y_tr).most_common():
        print(f"  {cls:<13} {cnt:>6} ({100*cnt/len(y_tr):.1f}%)")

    # Pick classifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=args.seed))])
    elif args.model == "tree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                      random_state=args.seed)
    elif args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, max_depth=12,
                                      class_weight="balanced",
                                      random_state=args.seed, n_jobs=-1)
    elif args.model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8,
                                              random_state=args.seed)

    print(f"\nfitting {args.model} ...")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    raw_acc = (y_pred == y_te).mean()
    kappa = cohens_kappa(list(y_te), list(y_pred))

    print(f"\n=== {args.model} on {rd.name} ===")
    print(f"raw match rate:    {raw_acc:.3f}  ({(y_pred==y_te).sum()}/{len(y_te)})")
    print(f"Cohen's kappa:     {kappa:.3f}")
    print(f"  > 0.80 strong; 0.60-0.80 substantial; 0.40-0.60 moderate; <0.40 weak")

    buckets = ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR")

    print("\n=== Test-set distribution ===")
    print(f"{'bucket':<13} {'gold':>8} {'cheap':>8}")
    for b in buckets:
        gn = (y_te == b).sum()
        cn = (y_pred == b).sum()
        print(f"{b:<13} {gn:>8} {cn:>8}")

    print("\n=== Per-class precision/recall (gold = truth) ===")
    print(f"{'bucket':<13} {'precision':>10} {'recall':>8} {'f1':>6} {'n_gold':>8}")
    for b in buckets:
        tp = ((y_te == b) & (y_pred == b)).sum()
        fp = ((y_te != b) & (y_pred == b)).sum()
        fn = ((y_te == b) & (y_pred != b)).sum()
        n_gold = (y_te == b).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2*p*r / max(p + r, 1e-9)
        print(f"{b:<13} {p:>10.3f} {r:>8.3f} {f1:>6.3f} {n_gold:>8}")

    print("\n=== Confusion (rows=gold, cols=cheap) ===")
    header = "".join(f"{b[:5]:>8}" for b in buckets)
    print(f"{'':<14}{header}")
    for gb in buckets:
        row = [str(((y_te == gb) & (y_pred == cb)).sum()) for cb in buckets]
        print(f"{gb:<14}" + "".join(f"{v:>8}" for v in row))

    # Feature importance / coefficients (if available)
    print("\n=== Feature importance ===")
    try:
        if args.model == "logreg":
            coef = clf.named_steps["clf"].coef_  # (n_classes, n_features)
            print("(per-class L2 coefficient magnitude, after scaling)")
            classes = clf.named_steps["clf"].classes_
            for c_i, cls in enumerate(classes):
                top = np.argsort(-np.abs(coef[c_i]))[:6]
                imps = ", ".join(f"{names[j]}: {coef[c_i][j]:+.2f}" for j in top)
                print(f"  {cls:<13} {imps}")
        elif args.model in ("tree", "rf", "gbm"):
            importances = (clf.feature_importances_ if args.model in ("tree", "rf")
                          else clf.feature_importances_)
            ranked = sorted(zip(names, importances), key=lambda x: -x[1])
            for name, imp in ranked[:10]:
                print(f"  {name:<25} {imp:.4f}")
    except Exception as exc:
        print(f"  (couldn't extract: {exc})")

    # Save model if requested
    if args.out_classifier:
        import joblib
        joblib.dump({"model": clf, "feature_names": names,
                     "test_kappa": kappa, "test_acc": raw_acc},
                    args.out_classifier)
        print(f"\nsaved fitted model to {args.out_classifier}")


if __name__ == "__main__":
    main()
