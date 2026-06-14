"""Compare coherence-label distributions across the ceiling-experiment models
(64d / 96d / 128d), with Wilson CIs on each fraction so we don't over-read
small differences (plan-sae-validation.md V1).

Reads feature_labels_sample512.json from each run dir; reports COHERENT,
THEMATIC, GENERIC, POLYSEMANTIC, UNCLEAR fractions and the COH+THM "useful"
fraction with 95% Wilson intervals. Answers: does the higher-dimension
feature surplus consist of coherent features, or long-tail noise?
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.stats_utils import proportion_ci  # noqa: E402

CLASSES = ["COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"]


def load_counts(run_dir: Path):
    f = run_dir / "feature_labels_sample512.json"
    d = json.loads(f.read_text())
    labels = d.get("labels", d.get("features", {}))
    counts = {c: 0 for c in CLASSES}
    for v in labels.values():
        j = (v.get("judgment") or v.get("label") or "UNCLEAR").upper()
        counts[j] = counts.get(j, 0) + 1
    return counts, sum(counts.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label:run_dir pairs, e.g. 64d:experiments/results/...")
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        tag, rd = spec.split(":", 1)
        counts, n = load_counts(Path(rd))
        rows.append((tag, counts, n))

    print(f"{'model':>10} {'n':>4}  " + "  ".join(f"{c[:4]:>13}" for c in CLASSES)
          + f"  {'COH+THM':>15}")
    for tag, counts, n in rows:
        cells = []
        for c in CLASSES:
            p, lo, hi = proportion_ci(counts[c], n)
            cells.append(f"{100*p:4.1f}[{100*lo:4.1f},{100*hi:4.1f}]")
        useful = counts["COHERENT"] + counts["THEMATIC"]
        up, ulo, uhi = proportion_ci(useful, n)
        print(f"{tag:>10} {n:>4}  " + "  ".join(cells)
              + f"  {100*up:4.1f}[{100*ulo:4.1f},{100*uhi:4.1f}]")

    print("\nInterpretation: compare COH+THM intervals. Overlapping intervals = "
          "no significant difference in feature quality at this sample size; the "
          "extra features at higher dim are then a real surplus of useful "
          "features (count grows, quality holds) rather than noise.")


if __name__ == "__main__":
    main()
