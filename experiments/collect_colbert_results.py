"""Collect existing ColBERT 17b/r/s eval results into the cross-family format.

No new training. Pulls:
- training metrics (results.json)
- topic cohesion (feature_topic_cohesion_scores.json)
- LLM rigor labels (feature_labels_sample512.json)
- TREC-COVID retrieval (/data/retrieval_treccovid_*.json)

Writes /data/colbert_cross_family_summary.json with a row per ColBERT
SAE matching the schema used for jina/MiniLM in
/data/jina_pipeline_stage_h_summary.json.

Usage:
    python -m experiments.collect_colbert_results
"""
import json
from pathlib import Path
from collections import Counter


COLBERT_TAGS = {
    "17b": "colbert_mxbai_phase17b_otf_32K_saelens_2B_*",
    "17r": "colbert_mxbai_phase17r_otf_131K_oldrecipe_replay4_2B_*",
    "17s": "colbert_mxbai_phase17s_otf_131K_oldrecipe_replay4_4B_*",
}


def load_treccovid_map():
    """Merge the two TREC-COVID files (17b vs 17r/17s) into {run_name: nDCG}."""
    out = {}
    for f in [Path("/data/retrieval_treccovid_old_vs_saelens.json"),
              Path("/data/retrieval_treccovid_17r_17s.json")]:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for r in d.get("results", []):
            name = r.get("run", "")
            out[name] = {
                "ndcg@10": r.get("ndcg@10"),
                "mrr@10": r.get("mrr@10"),
                "spearman_vs_raw": r.get("spearman_vs_raw"),
            }
    return out


def collect_one(tag: str, run_pat: str, treccovid: dict) -> dict:
    matches = sorted(Path("/home/enjalot/code/latent-sae/experiments/results").glob(run_pat),
                     reverse=True)
    matches = [m for m in matches if "folded" not in m.name]
    if not matches:
        return {"missing": True}
    run = matches[0]
    s = {"run": run.name}

    res = run / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        try:
            s["training"] = {
                "n_samples": r["data"].get("n_samples"),
                "throughput_samples_sec": r["timing"].get("avg_throughput_samples_sec"),
                "wall_seconds": r["timing"].get("total_training_time_s"),
                "n_params": r["model"].get("n_params"),
                "num_latents": r["model"].get("num_latents"),
                "k": r["config"].get("sae", {}).get("k"),
            }
        except Exception:
            s["training"] = {}

    coh = run / "feature_topic_cohesion_scores.json"
    if coh.exists():
        c = json.loads(coh.read_text())
        per = c.get("per_feature", [])
        gaps = [r.get("gap", 0.0) for r in per]
        s["cohesion"] = {
            "n_scored": len(per),
            "mean_gap": (sum(gaps) / len(gaps)) if gaps else None,
            "frac_gap_above_0.05": (sum(1 for g in gaps if g > 0.05) / len(gaps)) if gaps else None,
            "overall_within_cos": c.get("overall_within_cos"),
            "overall_baseline_cos": c.get("overall_baseline_cos"),
            "overall_auc": c.get("overall_auc"),
            "encoder": c.get("encoder"),
        }

    for cand in ["feature_labels_sample512.json", "feature_labels.json"]:
        lp = run / cand
        if lp.exists():
            ld = json.loads(lp.read_text())
            counts = ld.get("counts", {})
            if not counts and "labels" in ld:
                counts = Counter(v.get("judgment") for v in ld["labels"].values())
            s[f"sample_labels_{cand.replace('.json','')}"] = {
                "counts": dict(counts), "model": ld.get("model"),
                "n_features": len(ld.get("labels", {})),
            }
            break

    s["treccovid"] = treccovid.get(run.name, {})
    s["raw_treccovid"] = treccovid.get("raw_colbert", {})
    return s


def main():
    treccovid = load_treccovid_map()
    print(f"loaded {len(treccovid)} TREC-COVID runs")

    summary = {}
    for tag, pat in COLBERT_TAGS.items():
        summary[tag] = collect_one(tag, pat, treccovid)
        print(f"\n=== {tag}: {summary[tag].get('run', 'MISSING')} ===")
        t = summary[tag].get("training", {})
        c = summary[tag].get("cohesion", {})
        rt = summary[tag].get("treccovid", {})
        rr = summary[tag].get("raw_treccovid", {})
        print(f"  latents={t.get('num_latents')}  k={t.get('k')}  n_samples={t.get('n_samples')}")
        if c:
            print(f"  cohesion: gap={c.get('mean_gap', -1):.3f}  "
                  f"frac>0.05={c.get('frac_gap_above_0.05', 0)*100:.1f}%  "
                  f"AUC={c.get('overall_auc', 0):.3f}")
        if rt:
            print(f"  TREC-COVID: SAE-recon nDCG@10={rt.get('ndcg@10'):.4f}  "
                  f"raw={rr.get('ndcg@10'):.4f}  spearman={rt.get('spearman_vs_raw', 0):.3f}")
        for k in summary[tag]:
            if k.startswith("sample_labels_"):
                v = summary[tag][k]
                print(f"  {k}: {v['counts']} (n={v['n_features']}, model={v.get('model')})")

    out = Path("/data/colbert_cross_family_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
