"""
Compare results across SAE experiment runs.

Usage:
  python -m experiments.compare_results experiments/results/
  python -m experiments.compare_results experiments/results/ --filter "gpu_bench*"
  python -m experiments.compare_results experiments/results/ --sort throughput
  python -m experiments.compare_results experiments/results/ --csv results.csv
"""

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path


def load_results(dirs: list, filter_pattern: str = "") -> list:
    """Recursively find results.json files and load them."""
    results = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"Warning: {d} does not exist", file=sys.stderr)
            continue
        for results_file in p.rglob("results.json"):
            try:
                with open(results_file) as f:
                    r = json.load(f)
                r["_path"] = str(results_file.parent)
                name = r.get("config", {}).get("name", results_file.parent.name)
                if filter_pattern and not fnmatch.fnmatch(name, filter_pattern):
                    continue
                results.append(r)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: failed to load {results_file}: {e}", file=sys.stderr)
    return results


SORT_KEYS = {
    "name": lambda r: r.get("config", {}).get("name", ""),
    "time": lambda r: r.get("timing", {}).get("total_training_time_s", 0),
    "throughput": lambda r: r.get("timing", {}).get("avg_throughput_samples_sec", 0),
    "cost": lambda r: r.get("timing", {}).get("cost_estimate_usd", 0),
    "params": lambda r: r.get("model", {}).get("n_params", 0),
    "gpu": lambda r: r.get("infra", {}).get("gpu_type", ""),
}


def print_table(results: list, sort_by: str = "name"):
    """Print a formatted comparison table."""
    if not results:
        print("No results found.")
        return

    sort_fn = SORT_KEYS.get(sort_by, SORT_KEYS["name"])
    results.sort(key=sort_fn, reverse=(sort_by in ("throughput",)))

    # Header
    print(f"\n{'='*120}")
    print("SAE EXPERIMENT COMPARISON")
    print(f"{'='*120}")
    header = (
        f"{'Name':<36} "
        f"{'Type':<8} "
        f"{'Exp.F':>5} "
        f"{'K':>4} "
        f"{'Params':>10} "
        f"{'GPU':<10} "
        f"{'Time(s)':>8} "
        f"{'Samp/s':>8} "
        f"{'Mem(MB)':>8} "
        f"{'Cost($)':>8}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        if "error" in r:
            name = r.get("config", {}).get("name", "?")
            print(f"{name:<36} ERROR: {r['error']}")
            continue

        cfg = r.get("config", {})
        m = r.get("model", {})
        t = r.get("timing", {})
        infra = r.get("infra", {})

        print(
            f"{cfg.get('name', '?'):<36} "
            f"{m.get('sae_type', '?'):<8} "
            f"{m.get('expansion_factor', '?'):>5} "
            f"{m.get('k', '?'):>4} "
            f"{m.get('n_params', 0):>10_} "
            f"{infra.get('gpu_type', '?'):<10} "
            f"{t.get('total_training_time_s', 0):>8.1f} "
            f"{t.get('avg_throughput_samples_sec', 0):>8.0f} "
            f"{infra.get('gpu_memory_peak_mb', 0):>8.0f} "
            f"{t.get('cost_estimate_usd', 0):>8.4f}"
        )
    print()
    print(f"Total: {len(results)} runs")


def export_csv(results: list, path: str):
    """Export results to CSV."""
    if not results:
        return

    rows = []
    for r in results:
        cfg = r.get("config", {})
        m = r.get("model", {})
        t = r.get("timing", {})
        infra = r.get("infra", {})
        rows.append({
            "name": cfg.get("name", ""),
            "sae_type": m.get("sae_type", ""),
            "expansion_factor": m.get("expansion_factor", ""),
            "k": m.get("k", ""),
            "n_params": m.get("n_params", ""),
            "gpu_type": infra.get("gpu_type", ""),
            "n_samples": r.get("data", {}).get("n_samples", ""),
            "batch_size": cfg.get("train", {}).get("batch_size", ""),
            "training_time_s": t.get("total_training_time_s", ""),
            "throughput_samples_sec": t.get("avg_throughput_samples_sec", ""),
            "gpu_memory_peak_mb": infra.get("gpu_memory_peak_mb", ""),
            "cost_usd": t.get("cost_estimate_usd", ""),
            "config_hash": r.get("config_hash", ""),
            "timestamp": r.get("timestamp", ""),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare SAE experiment results")
    parser.add_argument("dirs", nargs="+", help="Directories to search for results.json")
    parser.add_argument("--filter", default="", help="Glob pattern to filter by experiment name")
    parser.add_argument("--sort", default="name", choices=list(SORT_KEYS.keys()), help="Sort by column")
    parser.add_argument("--csv", default="", help="Export results to CSV file")
    args = parser.parse_args()

    results = load_results(args.dirs, args.filter)
    print_table(results, sort_by=args.sort)

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()
