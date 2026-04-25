#!/bin/bash
# After Phase 9 sweep finishes, run all eval tools on new SAEs and aggregate.
#
# Order of operations:
#   1. Extract feature activations (10K chunks) for every new SAE
#   2. SciFact MaxSim retrieval on all new SAEs
#   3. Domain probe on ALL SAEs (old + new) — consistent cross-SAE view
#   4. Random-sample labeling (256 features) on all new SAEs with qwen2.5:7b
#   5. Pick top-2 by (retrieval AND coherent%), full-label with qwen2.5:32b
#   6. Autointerp prediction score on the 2 fully-labeled winners
#   7. Final summarize_all table
set -euo pipefail

cd /home/enjalot/code/latent-sae
PY=/home/enjalot/code/latent-data-modal/.venv/bin/python

# Patterns for Phase 9 runs
PHASE9_GLOB='experiments/results/colbert_mxbai_phase9*'

echo "=============================="
echo "1. Extract feature activations"
echo "=============================="
for run in $PHASE9_GLOB; do
  if [[ -f "$run/feature_activations.json" ]]; then continue; fi
  echo "-- $run"
  $PY -m experiments.extract_feature_activations \
    --sae-dir "$run" --dataset fineweb --n-chunks 10000 --top-n 20 \
    2>&1 | grep -E "run:|live features|wrote"
done

echo
echo "============================="
echo "2. SciFact MaxSim on new SAEs"
echo "============================="
# Collect every Phase 9 run dir, build --sae-dir args
SAE_ARGS=()
for run in $PHASE9_GLOB; do
  SAE_ARGS+=(--sae-dir "$run")
done
$PY -m experiments.eval_colbert_retrieval --dataset scifact "${SAE_ARGS[@]}" 2>&1 | tail -30

echo
echo "==============================="
echo "3. Domain probe on ALL labeled"
echo "==============================="
ALL_ARGS=()
# New Phase 9
for run in $PHASE9_GLOB; do ALL_ARGS+=(--sae-dir "$run"); done
# Selected prior runs for baseline comparison
for run in experiments/results/colbert_mxbai_phase4__k8_expansion_factor2_* \
           experiments/results/colbert_mxbai_phase4__k16_expansion_factor2_* \
           experiments/results/colbert_mxbai_phase4__k24_expansion_factor2_* \
           experiments/results/colbert_mxbai_phase5__k32_* \
           experiments/results/colbert_mxbai_phase1__k32_expansion_factor32_*; do
  ALL_ARGS+=(--sae-dir "$run")
done
$PY -m experiments.domain_probe "${ALL_ARGS[@]}" --n-chunks-per-domain 20 --top-k-features 12 2>&1 | tail -30

echo
echo "=========================================="
echo "4. Sample-label all new SAEs with qwen2.5:7b"
echo "=========================================="
for run in $PHASE9_GLOB; do
  if [[ -f "$run/feature_labels_sample256.json" ]]; then continue; fi
  echo "-- $run"
  $PY -m experiments.label_features_ollama \
    --input "$run/feature_activations.json" \
    --model qwen2.5:7b-instruct-q4_K_M \
    --sample-random 256 \
    2>&1 | tail -10
done

echo
echo "==============================="
echo "5. Rank and pick top-2 for full label"
echo "==============================="
$PY -m experiments.summarize_all --pattern "colbert_mxbai_phase9*" 2>&1 | tail -20
# Human-pickable; scripted selection by (ndcg + coherent%)
$PY - <<'PYEOF'
import json, glob
from pathlib import Path

rows = []
for rd in glob.glob("experiments/results/colbert_mxbai_phase9*"):
    rd = Path(rd)
    metrics = {}
    p = rd / "metrics.json"
    if p.exists(): metrics = json.loads(p.read_text())
    # retrieval
    retr = json.loads(Path("/data/embeddings/beir/scifact-mxbai-edge-32m/retrieval_results.json").read_text())
    ndcg = next((r["ndcg@10"] for r in retr["results"] if r.get("run") == rd.name), None)
    # sample labels
    sample = rd / "feature_labels_sample256.json"
    if not sample.exists() or ndcg is None: continue
    labels = json.loads(sample.read_text())
    total = sum(labels["counts"].values()) or 1
    coh = labels["counts"].get("COHERENT", 0) / total
    thm = labels["counts"].get("THEMATIC", 0) / total
    score = ndcg + coh + 0.5 * thm  # heuristic: retrieval + coherence first, thematic as tiebreaker
    rows.append((score, ndcg, coh, thm, rd.name))

rows.sort(reverse=True)
print("top candidates:")
for s, n, c, t, name in rows[:5]:
    print(f"  score={s:.3f}  nDCG={n:.3f}  COH%={c:.3f}  THM%={t:.3f}  {name}")
# Pick top 2 as winners; save names to file
top2 = [r[4] for r in rows[:2]]
Path("/data/phase9_top2.txt").write_text("\n".join(top2) + "\n")  # trailing newline so shell `while read` doesn't drop last line
print(f"\ntop-2 winners -> /data/phase9_top2.txt")
PYEOF

echo
echo "========================================="
echo "6. Full-label top-2 with qwen2.5:32b"
echo "========================================="
while IFS= read -r run_name; do
  run="experiments/results/$run_name"
  if [[ -f "$run/feature_labels.json" ]]; then echo "skip $run"; continue; fi
  echo "-- $run"
  $PY -m experiments.label_features_ollama \
    --input "$run/feature_activations.json" \
    --model qwen2.5:32b-instruct-q4_K_M \
    2>&1 | tail -10
done < /data/phase9_top2.txt

echo
echo "========================================="
echo "7. Autointerp prediction score on winners"
echo "========================================="
while IFS= read -r run_name; do
  run="experiments/results/$run_name"
  [[ -f "$run/feature_labels.json" ]] || continue
  echo "-- $run"
  $PY -m experiments.autointerp_predict \
    --run-dir "$run" \
    --model qwen2.5:7b-instruct-q4_K_M \
    --n-features 64 \
    2>&1 | tail -10
done < /data/phase9_top2.txt

echo
echo "========================================="
echo "8. Final aggregate"
echo "========================================="
$PY -m experiments.summarize_all --pattern "colbert_mxbai_*" 2>&1 | tail -60

echo
echo "========================================="
echo "9. Consolidated Markdown report"
echo "========================================="
$PY -m experiments.final_report

echo
echo "=== PIPELINE DONE ==="
