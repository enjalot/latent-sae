"""Generate a consolidated Markdown report covering all phases' findings.

Pulls data from:
  - experiments/results/*/results.json, metrics.json, feature_labels*.json
  - /data/embeddings/beir/scifact-mxbai-edge-32m/retrieval_results.json
  - /data/embeddings/beir/domain_probe.json

Output: experiments/results/FINAL_REPORT.md
"""
import json
from pathlib import Path


RESULTS = Path("experiments/results")
RETR = Path("/data/embeddings/beir/scifact-mxbai-edge-32m/retrieval_results.json")
DOMAIN = Path("/data/embeddings/beir/domain_probe.json")
OUT = RESULTS / "FINAL_REPORT.md"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def main():
    retrieval = load(RETR) or {"results": []}
    retr_by_run = {r["run"]: r for r in retrieval["results"]}

    rows = []
    for rd in sorted(RESULTS.glob("colbert_mxbai_*")):
        if not rd.is_dir():
            continue
        res = load(rd / "results.json") or {}
        met = load(rd / "metrics.json") or {}
        # prefer full labels; fall back to sample labels
        lab = (load(rd / "feature_labels.json")
               or load(rd / "feature_labels_sample256.json")
               or {})
        # Prefer rigorous prediction scores when present (graded over all
        # COHERENT features, not a 64-feature sample).
        pred = (load(rd / "feature_prediction_scores_rigorous.json")
                or load(rd / "feature_prediction_scores.json")
                or {})

        model = res.get("model", {})
        infra = res.get("infra", {})
        counts = lab.get("counts", {}) if lab else {}
        total = sum(counts.values()) or None
        def pct(k):
            return 100 * counts.get(k, 0) / total if total else None

        retr = retr_by_run.get(rd.name, {})
        rows.append({
            "run": rd.name,
            "sae_type": model.get("sae_type"),
            "k": model.get("k"),
            "exp": model.get("expansion_factor"),
            "num_latents": met.get("num_latents"),
            "fvu": met.get("fvu"),
            "l0": met.get("l0"),
            "dead_pct": met.get("dead_pct"),
            "mmcs": met.get("mmcs"),
            "ndcg10": retr.get("ndcg@10"),
            "mrr10": retr.get("mrr@10"),
            "spearman": retr.get("spearman_vs_raw"),
            "label_source": ("full" if (rd / "feature_labels.json").exists()
                             else "sample" if (rd / "feature_labels_sample256.json").exists()
                             else None),
            "coh_pct": pct("COHERENT"),
            "thm_pct": pct("THEMATIC"),
            "gen_pct": pct("GENERIC"),
            "pol_pct": pct("POLYSEMANTIC"),
            "n_live": lab.get("n_live_features"),
            "predict_acc": pred.get("overall_accuracy"),
            "predict_per_label": pred.get("per_label_accuracy"),
        })

    def f(x, n=4):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) and x is not None else "-"
    def p(x):
        return f"{x:.1f}%" if isinstance(x, (int, float)) and x is not None else "-"

    md = []
    md.append("# Consolidated SAE Eval Report\n")
    md.append("Eval dimensions: FVU / L0 / dead% / MMCS (proxies) • SciFact MaxSim nDCG@10 and ρ_rank (retrieval) • COH/THM/GEN/POL % (interpretability) • Prediction accuracy (autointerp faithfulness).\n")
    md.append("\n## All runs\n")
    md.append("| run | arch | k | exp | latents | FVU | dead% | MMCS | nDCG | ρ_rank | COH% | THM% | GEN% | POL% | n_live | pred_acc |")
    md.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        md.append("| {run} | {st} | {k} | {exp} | {lat} | {fvu} | {dd} | {mm} | {nd} | {rho} | {coh} | {thm} | {gen} | {pol} | {nl} | {pa} |".format(
            run=r["run"][:60], st=r["sae_type"] or "?", k=r["k"] or "?", exp=r["exp"] or "?",
            lat=r["num_latents"] or "?",
            fvu=f(r["fvu"]), dd=p(100*r["dead_pct"]) if r["dead_pct"] is not None else "-",
            mm=f(r["mmcs"], 3), nd=f(r["ndcg10"]), rho=f(r.get("spearman"), 3),
            coh=p(r["coh_pct"]), thm=p(r["thm_pct"]), gen=p(r["gen_pct"]), pol=p(r["pol_pct"]),
            nl=r["n_live"] or "-",
            pa=f(r["predict_acc"], 3)))

    # Highlight rows with predict_acc available
    md.append("\n## Runs with autointerp prediction score (labels grounded)\n")
    graded = [r for r in rows if r.get("predict_acc") is not None]
    for r in graded:
        md.append(f"\n### {r['run']}\n")
        md.append(f"- overall pred accuracy: **{r['predict_acc']:.3f}**")
        ppl = r.get("predict_per_label") or {}
        for lbl, acc in ppl.items():
            md.append(f"- {lbl}: {acc:.3f}")

    # Coherent-feature count estimates (both raw and prediction-validated)
    md.append("\n## Interpretable-feature counts (estimated)\n")
    md.append("Multiplies sampled COH% × n_live_features. Entries with full labels")
    md.append("(feature_labels.json) are exact; others extrapolate from the sample.\n")
    md.append("| run | arch | latents | n_live | COH% | COH count |")
    md.append("|---|---|--:|--:|--:|--:|")
    est = []
    for i, r in enumerate(rows):
        nl = r.get("n_live")
        coh = r.get("coh_pct")
        if nl and coh is not None:
            coh_count = int(round(nl * coh / 100))
            est.append((coh_count, i, r))  # index as tiebreaker to avoid dict compare
    est.sort(key=lambda t: -t[0])
    for count, _idx, r in est[:15]:
        md.append(f"| {r['run'][:55]} | {r['sae_type']} | {r['num_latents']} | {r['n_live']} | {r['coh_pct']:.1f}% | **{count}** |")

    # Top by each criterion
    md.append("\n## Pareto picks\n")
    def nonnull(key): return [r for r in rows if r.get(key) is not None]
    for label, key, desc in [
        ("Highest nDCG@10", "ndcg10", "best retrieval"),
        ("Highest COH%", "coh_pct", "most strictly-coherent features"),
        ("Highest (nDCG + COH%)", None, "joint pick"),
        ("Most live features", "n_live", "capacity"),
    ]:
        if key is None:
            ranked = sorted([r for r in rows if r.get("ndcg10") is not None and r.get("coh_pct") is not None],
                            key=lambda r: -(r["ndcg10"] + r["coh_pct"]/100))
        else:
            ranked = sorted(nonnull(key), key=lambda r: -r[key])
        md.append(f"\n**{label}** ({desc}):")
        for r in ranked[:5]:
            md.append(f"- `{r['run']}` — nDCG={f(r['ndcg10'])}  COH={p(r['coh_pct'])}  n_live={r['n_live'] or '-'}  arch={r['sae_type']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
