"""Topic-cohesion scorer (label-free).

Asks a different question than the other three scorers: does this feature
fire on a coherent TOPIC, regardless of whether its activating tokens are
heterogeneous BPE fragments?

For each feature with enough activations:
  1. Take the top-N activating windows.
  2. Embed each window's full text with a sentence encoder
     (all-MiniLM-L6-v2, 384-dim, sentence-trained — less hypersphere
     concentration than mean-pooled 64-dim ColBERT, so cosines have
     much more dynamic range here).
  3. Compute the mean pairwise cosine similarity within the feature's
     own windows = within_cos.
  4. Compute a baseline: pairwise cosines between this feature's
     windows and a shared pool of random-feature windows = baseline_cos.
  5. Per-feature score = within_cos - baseline_cos (positive means more
     topically coherent than chance).

Also reports the per-feature AUC of (within similarity ranks vs across-
feature similarity ranks) so we can read the score against the embedsim
scorer on equal footing.

Crucially this is independent of any LM-written description: it scores
the feature's activation pattern directly, not the human-readable label.
That is the diagnostic the 17i revision experiment told us we needed.

Usage:
    python -m experiments.autointerp_topic_cohesion \\
        --run-dir experiments/results/<run> \\
        [--n-features 0]   # 0 = score every live feature with enough hits
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SENTENCE_MODEL_ID = "all-MiniLM-L6-v2"
SENTENCE_CACHE = "/data/hf/sentence-transformers"
TOKEN_MARKER_RE = re.compile(r"<<|>>")


def strip_markers(window: str) -> str:
    return TOKEN_MARKER_RE.sub("", window).strip()


def encode_sentences(texts: list[str], model, batch_size: int = 128) -> np.ndarray:
    """Encode texts with sentence-transformers, return l2-normalized (n, dim)."""
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-features", type=int, default=0,
                    help="0 = score every live feature with >= --min-hits hits")
    ap.add_argument("--min-hits", type=int, default=10)
    ap.add_argument("--max-hits", type=int, default=20,
                    help="cap windows per feature for cohesion calc")
    ap.add_argument("--baseline-pool-size", type=int, default=200,
                    help="size of random pooled vectors for baseline cohesion")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    acts_data = json.loads((run_dir / "feature_activations.json").read_text())

    # Load labels if available — for per-class aggregation
    labels = None
    for cand in ["feature_labels.json", "feature_labels_sample512.json"]:
        p = run_dir / cand
        if p.exists():
            labels = json.loads(p.read_text())
            print(f"loaded labels from {cand}: {len(labels.get('labels', {}))} features")
            break

    # Pick features to score. Prefer labeled features when labels are
    # available — otherwise we can't aggregate per class.
    rng = random.Random(args.seed)
    if labels and labels.get("labels"):
        eligible = [fid for fid in labels["labels"]
                    if len(acts_data["features"].get(fid, [])) >= args.min_hits]
        print(f"using {len(eligible)} labeled features (with >= {args.min_hits} hits)")
    else:
        eligible = [fid for fid, hits in acts_data["features"].items()
                    if len(hits) >= args.min_hits]
    if args.n_features > 0 and len(eligible) > args.n_features:
        rng.shuffle(eligible)
        eligible = eligible[: args.n_features]
    eligible.sort(key=int)
    print(f"scoring {len(eligible)} features")

    # Build the FLAT list of all windows we need to encode (per-feature + baseline)
    # This single batch encode dominates cost.
    feat_window_slices = {}  # fid -> (start, end) into vectors array
    flat_texts = []
    for fid_s in eligible:
        hits = acts_data["features"][fid_s]
        windows = [strip_markers(h.get("window", "")) for h in hits[: args.max_hits]]
        windows = [w for w in windows if w]
        s = len(flat_texts)
        flat_texts.extend(windows)
        e = len(flat_texts)
        feat_window_slices[fid_s] = (s, e)

    # Random-baseline pool: sample windows from random eligible features
    rng_b = random.Random(args.seed + 1)
    pool_fids = rng_b.sample(eligible, k=min(args.baseline_pool_size, len(eligible)))
    pool_start = len(flat_texts)
    for fid_s in pool_fids:
        # Take ONE window from each pool feature so the baseline is a
        # random scatter across features rather than concentrated in a
        # few features' window sets.
        hits = acts_data["features"][fid_s]
        if not hits:
            continue
        idx = rng_b.randrange(min(args.max_hits, len(hits)))
        w = strip_markers(hits[idx].get("window", ""))
        if w:
            flat_texts.append(w)
    pool_end = len(flat_texts)

    print(f"encoding {len(flat_texts)} total windows (per-feature + baseline pool)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SENTENCE_MODEL_ID, cache_folder=SENTENCE_CACHE,
                                device=args.device)
    t0 = time.monotonic()
    vecs = encode_sentences(flat_texts, model)
    print(f"encoded in {time.monotonic() - t0:.1f}s; shape={vecs.shape}")

    pool_vecs = vecs[pool_start:pool_end]  # (P, dim)
    print(f"baseline pool: {pool_vecs.shape[0]} windows")

    per_feature = []
    by_label = {}
    for fid_s in eligible:
        s, e = feat_window_slices[fid_s]
        if e - s < 3:
            continue
        own = vecs[s:e]                                # (n, dim)
        # Within-feature: mean off-diagonal cos
        sim = own @ own.T                              # (n, n)
        n = own.shape[0]
        # exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        within = float(sim[mask].mean())
        # Baseline: own vs pool
        cross = own @ pool_vecs.T                      # (n, P)
        baseline = float(cross.mean())
        gap = within - baseline

        # AUC: rank of within-pair sims vs cross-pair sims
        within_pairs = sim[mask]
        cross_pairs = cross.flatten()
        # Mann–Whitney U → AUC
        all_vals = np.concatenate([within_pairs, cross_pairs])
        ranks = np.argsort(np.argsort(all_vals)) + 1
        n_w = len(within_pairs); n_c = len(cross_pairs)
        u = ranks[:n_w].sum() - n_w * (n_w + 1) / 2
        auc = float(u / (n_w * n_c))

        rec = {
            "feature": fid_s, "n_windows": n,
            "within_cos": within,
            "baseline_cos": baseline,
            "gap": gap,
            "auc": auc,
        }
        if labels and fid_s in labels.get("labels", {}):
            rec["label"] = labels["labels"][fid_s]["judgment"]
            by_label.setdefault(rec["label"], []).append(rec)
        per_feature.append(rec)

    overall_within = float(np.mean([r["within_cos"] for r in per_feature]))
    overall_baseline = float(np.mean([r["baseline_cos"] for r in per_feature]))
    overall_gap = overall_within - overall_baseline
    overall_auc = float(np.mean([r["auc"] for r in per_feature]))

    report = {
        "run": run_dir.name,
        "scorer": "topic_cohesion_minilm",
        "encoder": SENTENCE_MODEL_ID,
        "n_features_scored": len(per_feature),
        "overall_within_cos": overall_within,
        "overall_baseline_cos": overall_baseline,
        "overall_gap": overall_gap,
        "overall_auc": overall_auc,
        "per_label_auc": {k: float(np.mean([r["auc"] for r in v])) for k, v in by_label.items()},
        "per_label_gap": {k: float(np.mean([r["gap"] for r in v])) for k, v in by_label.items()},
        "per_label_n": {k: len(v) for k, v in by_label.items()},
        "per_feature": per_feature,
    }
    out = run_dir / "feature_topic_cohesion_scores.json"
    out.write_text(json.dumps(report, indent=2))

    print()
    print(f"=== {run_dir.name} ===")
    print(f"overall: within_cos={overall_within:.3f}  baseline_cos={overall_baseline:.3f}  gap={overall_gap:+.3f}  AUC={overall_auc:.3f}")
    if by_label:
        print(f"{'label':<13} {'n':>5} {'within':>8} {'gap':>8} {'AUC':>8}")
        for lbl in ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"):
            if lbl not in by_label: continue
            rs = by_label[lbl]
            print(f"{lbl:<13} {len(rs):>5} "
                  f"{np.mean([r['within_cos'] for r in rs]):>8.3f} "
                  f"{np.mean([r['gap'] for r in rs]):>+8.3f} "
                  f"{np.mean([r['auc'] for r in rs]):>8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
