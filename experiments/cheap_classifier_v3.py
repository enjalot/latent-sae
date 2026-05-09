"""Cheap classifier v3 — adds TF-IDF text features and a binary
'interpretable vs not' framing for downstream filtering use cases.

Two predictions per feature:
  1. Five-class bucket (same as 32b judge)
  2. Binary 'interpretable' (COH or THM) — useful for latent-taxonomy
     filtering even if the 5-class kappa is moderate

Usage:
    python -m experiments.cheap_classifier_v3 --run-dir <17r dir>
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# Reuse text patterns from v2
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


def numeric_features(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "cohesion_gap", "cohesion_auc",
        "embedsim_auc", "embedsim_mean_pos", "embedsim_mean_neg", "embedsim_pos_neg_gap",
        "n_hits", "n_hits_log",
        "desc_len", "desc_word_count", "desc_word_count_log",
        "has_poly_word", "has_gen_word", "has_uncl_word",
        "has_specific_concept",
    ]
    rows = []
    for r in records:
        desc = r.get("description", "")
        words = desc.split()
        wlen = len(words)
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
    return np.asarray(rows, dtype=np.float32), feature_names


def assemble_records(run_dir: Path) -> list[dict]:
    labels_path = (run_dir / "feature_labels.json").resolve()
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


def report_5class(y_te, y_pred, header: str):
    buckets = ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR")
    raw = (y_pred == y_te).mean()
    kappa = cohens_kappa(list(y_te), list(y_pred))
    print(f"\n=== {header} ===")
    print(f"raw match: {raw:.3f}    kappa: {kappa:.3f}")
    print(f"{'bucket':<13} {'gold':>6} {'cheap':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for b in buckets:
        gn = (y_te == b).sum()
        cn = (y_pred == b).sum()
        tp = ((y_te == b) & (y_pred == b)).sum()
        fp = ((y_te != b) & (y_pred == b)).sum()
        fn = ((y_te == b) & (y_pred != b)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2*p*r / max(p + r, 1e-9)
        print(f"{b:<13} {gn:>6} {cn:>6} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
    return raw, kappa


def report_binary(y_te, y_pred, header: str):
    """Treat COH+THM as 'interpretable=1', else 0."""
    interp = lambda x: int(x in ("COHERENT", "THEMATIC"))
    yt = np.array([interp(x) for x in y_te])
    yp = np.array([interp(x) for x in y_pred])
    raw = (yp == yt).mean()
    kappa = cohens_kappa(list(yt), list(yp))
    tp = ((yt == 1) & (yp == 1)).sum()
    fp = ((yt == 0) & (yp == 1)).sum()
    fn = ((yt == 1) & (yp == 0)).sum()
    tn = ((yt == 0) & (yp == 0)).sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2*p*r / max(p + r, 1e-9)
    print(f"\n=== {header} (interpretable yes/no) ===")
    print(f"raw match: {raw:.3f}    kappa: {kappa:.3f}")
    print(f"interpretable: P={p:.3f}  R={r:.3f}  F1={f1:.3f}    "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return raw, kappa, p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-classifier", default=None)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    print(f"loading from: {rd}")
    records = assemble_records(rd)
    print(f"{len(records)} feature records")

    # Numeric features
    X_num, num_names = numeric_features(records)
    descs = [r["description"] for r in records]
    y = np.array([r["judgment"] for r in records])

    # TF-IDF on descriptions
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=600, ngram_range=(1, 2),
                            stop_words="english", min_df=5, max_df=0.7,
                            lowercase=True)
    X_text = tfidf.fit_transform(descs)
    print(f"tfidf shape: {X_text.shape}")

    # Concatenate numeric + tfidf as sparse
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([csr_matrix(X_num), X_text])
    print(f"combined feature dim: {X_combined.shape[1]}")

    # Split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_test = int(len(idx) * args.test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_tr = X_combined[train_idx]; X_te = X_combined[test_idx]
    y_tr = y[train_idx]; y_te = y[test_idx]
    print(f"train: {len(y_tr)}, test: {len(y_te)}")

    # Train logreg with TF-IDF (sparse-friendly, fast)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=2.0,
                              class_weight="balanced",
                              n_jobs=-1, random_state=args.seed)
    print("\nfitting LogisticRegression with numeric + TF-IDF features...")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    raw5, k5 = report_5class(y_te, y_pred, "5-class on test")
    rawb, kb, pb, rb, f1b = report_binary(y_te, y_pred, "binary 'interpretable' on test")

    # Confusion (5-class)
    buckets = ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR")
    print("\n=== Confusion 5-class (rows=gold, cols=cheap) ===")
    header = "".join(f"{b[:5]:>8}" for b in buckets)
    print(f"{'':<14}{header}")
    for gb in buckets:
        row = [str(((y_te == gb) & (y_pred == cb)).sum()) for cb in buckets]
        print(f"{gb:<14}" + "".join(f"{v:>8}" for v in row))

    # Top tfidf coefficients per class (for interpretability + debugging)
    print("\n=== Top TF-IDF terms per class (model coefficients) ===")
    classes = clf.classes_
    feature_names_all = num_names + list(tfidf.get_feature_names_out())
    for ci, cls in enumerate(classes):
        coefs = clf.coef_[ci]
        # Rank tfidf features only (skip the first len(num_names))
        tf_coefs = coefs[len(num_names):]
        top_pos = np.argsort(-tf_coefs)[:8]
        terms = [tfidf.get_feature_names_out()[i] for i in top_pos]
        print(f"  {cls:<13} pos: {', '.join(terms)}")

    if args.out_classifier:
        import joblib
        joblib.dump({
            "logreg": clf, "tfidf": tfidf,
            "numeric_names": num_names,
            "test_kappa_5class": k5, "test_acc_5class": raw5,
            "test_kappa_binary": kb, "test_f1_binary": f1b,
        }, args.out_classifier)
        print(f"\nsaved fitted model to {args.out_classifier}")


if __name__ == "__main__":
    main()
