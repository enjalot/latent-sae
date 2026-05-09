"""Compare vllm vs ollama labeling on a fixed sample of features.

Loads two label files (ollama and vllm) produced from the SAME
feature_activations.json and the same --sample-random / seed.
Reports:

  1. Label distribution (counts per bucket, both backends).
  2. Per-feature bucket agreement (Cohen's kappa-style + raw match rate).
  3. Description similarity (mean Jaccard overlap of word sets).
  4. Throughput comparison (from each label file's metadata).

This is the gating test before committing to vllm for full 17r labeling:
if bucket agreement is high (>= 0.80 Cohen's kappa) and descriptions are
not wildly different, vllm AWQ is acceptable; if not, fall back to
ollama or investigate quality drop.

Usage:
    python -m experiments.validate_vllm_vs_ollama \\
        --ollama experiments/results/<run>/feature_labels_sample50.json \\
        --vllm   experiments/results/<run>/feature_labels_sample50_vllm.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def word_set(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text or "")}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cohens_kappa(items_a: list[str], items_b: list[str]) -> float:
    """Categorical agreement between two raters, chance-corrected."""
    cats = sorted(set(items_a) | set(items_b))
    n = len(items_a)
    obs_agreement = sum(1 for a, b in zip(items_a, items_b) if a == b) / n
    pa = Counter(items_a)
    pb = Counter(items_b)
    expected_agreement = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    if expected_agreement >= 1:
        return 1.0
    return (obs_agreement - expected_agreement) / (1 - expected_agreement)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", required=True, help="ollama-produced labels JSON")
    ap.add_argument("--vllm", required=True, help="vllm-produced labels JSON")
    args = ap.parse_args()

    o = json.loads(Path(args.ollama).read_text())
    v = json.loads(Path(args.vllm).read_text())

    o_lab = o["labels"]
    v_lab = v["labels"]
    common = sorted(set(o_lab) & set(v_lab), key=int)
    if not common:
        print("ERROR: no overlapping features between the two files")
        return
    print(f"Comparing {len(common)} features (ollama: {len(o_lab)}, vllm: {len(v_lab)})")

    # 1. Label distributions
    print("\n=== Label distribution ===")
    buckets = ("COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR")
    print(f"{'bucket':<13} {'ollama':>8} {'vllm':>8}")
    for b in buckets:
        oc = sum(1 for fid in common if o_lab[fid]["judgment"] == b)
        vc = sum(1 for fid in common if v_lab[fid]["judgment"] == b)
        print(f"{b:<13} {oc:>8} {vc:>8}")

    # 2. Per-feature bucket agreement
    print("\n=== Per-feature bucket agreement ===")
    o_judgments = [o_lab[fid]["judgment"] for fid in common]
    v_judgments = [v_lab[fid]["judgment"] for fid in common]
    raw_match = sum(1 for a, b in zip(o_judgments, v_judgments) if a == b) / len(common)
    kappa = cohens_kappa(o_judgments, v_judgments)
    print(f"raw match rate:   {raw_match:.3f}  ({sum(1 for a, b in zip(o_judgments, v_judgments) if a == b)}/{len(common)})")
    print(f"Cohen's kappa:    {kappa:.3f}")
    print(f"  > 0.80 → strong agreement")
    print(f"  0.60-0.80 → substantial")
    print(f"  < 0.60 → questionable, investigate")

    # Confusion matrix
    print("\nconfusion (rows=ollama, cols=vllm):")
    print(f"{'':<14}" + "".join(f"{b:>13}" for b in buckets))
    for ob in buckets:
        row = [ob]
        for vb in buckets:
            n = sum(1 for fid in common
                    if o_lab[fid]["judgment"] == ob and v_lab[fid]["judgment"] == vb)
            row.append(str(n))
        print(f"{row[0]:<14}" + "".join(f"{x:>13}" for x in row[1:]))

    # 3. Description similarity (word-set Jaccard)
    print("\n=== Description similarity (word-set Jaccard, words >=3 chars) ===")
    sims = []
    o_lens = []
    v_lens = []
    for fid in common:
        od = o_lab[fid]["description"]
        vd = v_lab[fid]["description"]
        sims.append(jaccard(word_set(od), word_set(vd)))
        o_lens.append(len(od))
        v_lens.append(len(vd))
    sims_sorted = sorted(sims)
    print(f"mean Jaccard:     {sum(sims)/len(sims):.3f}")
    print(f"median Jaccard:   {sims_sorted[len(sims)//2]:.3f}")
    print(f"min/max Jaccard:  {min(sims):.3f} / {max(sims):.3f}")
    print(f"avg desc length:  ollama={sum(o_lens)//len(o_lens)} chars, vllm={sum(v_lens)//len(v_lens)} chars")

    # Show the 5 most disagreeing examples
    print("\n=== 5 lowest-similarity examples ===")
    indexed = sorted(zip(sims, common), key=lambda x: x[0])
    for sim, fid in indexed[:5]:
        print(f"\n  feat {fid} (Jaccard={sim:.2f}, ollama={o_lab[fid]['judgment']}, vllm={v_lab[fid]['judgment']})")
        print(f"    ollama: {o_lab[fid]['description'][:160]}")
        print(f"    vllm:   {v_lab[fid]['description'][:160]}")

    # 4. Throughput from metadata if available
    print("\n=== Throughput ===")
    print(f"  ollama: backend not stamped (assume serial ~0.69 features/sec from prior measurements)")
    if "concurrency" in v:
        print(f"  vllm:   concurrency={v['concurrency']}, model={v['model']}")
    else:
        print(f"  vllm:   metadata missing concurrency field")


if __name__ == "__main__":
    main()
