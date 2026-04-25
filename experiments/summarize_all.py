"""Cross-reference all eval dimensions per SAE config.

Pulls:
  - config (k, expansion, sae_type) from run_dir/results.json
  - proxy metrics (FVU, dead%, MMCS) from run_dir/metrics.json
  - SciFact retrieval nDCG@10 from <cache>/retrieval_results.json
  - autointerp counts from run_dir/feature_labels.json

Emits a unified table to stdout (and optional CSV).
"""
import argparse
import csv
import json
from pathlib import Path


def load_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def run_row(run_dir: Path, retrieval: dict) -> dict:
    results = load_json(run_dir / "results.json") or {}
    metrics = load_json(run_dir / "metrics.json") or {}
    labels = load_json(run_dir / "feature_labels.json") or {}
    model = results.get("model", {})

    # Retrieval lookup
    ret_row = next((r for r in retrieval.get("results", [])
                    if r.get("run") == run_dir.name), None)
    ndcg = ret_row.get("ndcg@10") if ret_row else None

    counts = labels.get("counts", {}) if labels else {}
    total_labeled = sum(counts.values()) if counts else 0
    coh = counts.get("COHERENT", 0)
    thm = counts.get("THEMATIC", 0)
    gen = counts.get("GENERIC", 0)
    pol = counts.get("POLYSEMANTIC", 0)
    unc = counts.get("UNCLEAR", 0)

    def pct(x): return 100 * x / total_labeled if total_labeled else None

    return {
        "run": run_dir.name,
        "sae_type": model.get("sae_type", "?"),
        "k": model.get("k", "?"),
        "exp": model.get("expansion_factor", "?"),
        "num_latents": metrics.get("num_latents", "?"),
        "fvu": metrics.get("fvu"),
        "l0": metrics.get("l0"),
        "dead_pct": metrics.get("dead_pct"),
        "mmcs": metrics.get("mmcs"),
        "ndcg10": ndcg,
        "labeled": total_labeled,
        "coherent": coh, "coherent_pct": pct(coh),
        "thematic": thm, "thematic_pct": pct(thm),
        "generic": gen, "generic_pct": pct(gen),
        "polysemantic": pol, "polysemantic_pct": pct(pol),
        "unclear": unc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--pattern", default="colbert_mxbai_*")
    ap.add_argument("--retrieval",
                    default="/data/embeddings/beir/scifact-mxbai-edge-32m/retrieval_results.json")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    retrieval = load_json(Path(args.retrieval)) or {}
    rows = []
    for run_dir in sorted(Path(args.results_dir).glob(args.pattern)):
        if run_dir.is_dir():
            rows.append(run_row(run_dir, retrieval))

    # Print
    def fmt_pct(x): return f"{x:.1f}%" if x is not None else "-"
    def fmt_f(x, n=4): return f"{x:.{n}f}" if x is not None else "-"
    print(f"{'run':62}  {'k':>3} {'exp':>3} {'lat':>5}  "
          f"{'FVU':>7} {'dead%':>6} {'MMCS':>5}  "
          f"{'nDCG':>6}  {'coh%':>6} {'thm%':>6} {'gen%':>6} {'pol%':>6}")
    for r in rows:
        nm = r["run"][:60]
        print(f"{nm:62}  {str(r['k']):>3} {str(r['exp']):>3} {str(r['num_latents']):>5}  "
              f"{fmt_f(r['fvu']):>7} {fmt_pct(100*r['dead_pct'] if r['dead_pct'] is not None else None):>6} "
              f"{fmt_f(r['mmcs'],3):>5}  "
              f"{fmt_f(r['ndcg10']):>6}  "
              f"{fmt_pct(r['coherent_pct']):>6} {fmt_pct(r['thematic_pct']):>6} "
              f"{fmt_pct(r['generic_pct']):>6} {fmt_pct(r['polysemantic_pct']):>6}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
