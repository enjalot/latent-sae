#!/bin/bash
# Autonomous SAE autoresearch loop
# Usage: ./run_loop.sh [agent_id] [hours]
# Runs experiments in a loop, each ~2-3 min, for the specified duration.

AGENT_ID="${1:-1}"
HOURS="${2:-12}"
REPO_DIR="/Users/enjalot/code/latent-sae"
TRAIN_SCRIPT="experiments/autoresearch/train.py"
DATA_DIR="experiments/autoresearch/.data/train"
RESULTS_TSV="experiments/autoresearch/results_agent${AGENT_ID}.tsv"
LOG_DIR="experiments/autoresearch/logs"
DEVICE="mps"
VENV="$REPO_DIR/venv/bin/python"

export HF_HOME=/tmp/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/st

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

# Initialize results TSV if needed
if [ ! -f "$RESULTS_TSV" ]; then
    echo -e "timestamp\tcomposite\tclinc150\tscifact\tmmcs\tfvu\tactive\tstatus\tdescription" > "$RESULTS_TSV"
fi

END_TIME=$(($(date +%s) + HOURS * 3600))
EXP_NUM=0

echo "[Agent $AGENT_ID] Starting autoresearch loop for $HOURS hours"
echo "[Agent $AGENT_ID] End time: $(date -r $END_TIME)"

while [ $(date +%s) -lt $END_TIME ]; do
    EXP_NUM=$((EXP_NUM + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/agent${AGENT_ID}_exp${EXP_NUM}_${TIMESTAMP}.log"

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "[Agent $AGENT_ID] Experiment $EXP_NUM at $TIMESTAMP"
    echo "═══════════════════════════════════════════════════"

    # Run experiment
    $VENV "$TRAIN_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --d-in 384 \
        --device "$DEVICE" \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[Agent $AGENT_ID] CRASHED (exit $EXIT_CODE)"
        echo -e "${TIMESTAMP}\t0\t0\t0\t0\t0\t0\tcrash\texit_code_${EXIT_CODE}" >> "$RESULTS_TSV"
        # Wait a bit before retrying
        sleep 10
        continue
    fi

    # Extract results
    COMPOSITE=$(grep "^composite_score:" "$LOG_FILE" | awk '{print $2}')
    CLINC=$(grep "^clinc150_sparse:" "$LOG_FILE" | awk '{print $2}')
    SCIFACT=$(grep "^scifact_sparse:" "$LOG_FILE" | awk '{print $2}')
    MMCS=$(grep "^mmcs:" "$LOG_FILE" | awk '{print $2}')
    FVU=$(grep "^fvu:" "$LOG_FILE" | awk '{print $2}')
    ACTIVE=$(grep "^active_features:" "$LOG_FILE" | awk '{print $2}')

    if [ -z "$COMPOSITE" ]; then
        echo "[Agent $AGENT_ID] No results found in log"
        echo -e "${TIMESTAMP}\t0\t0\t0\t0\t0\t0\tno_results\tgrep_empty" >> "$RESULTS_TSV"
        continue
    fi

    echo "[Agent $AGENT_ID] composite=$COMPOSITE clinc=$CLINC scifact=$SCIFACT mmcs=$MMCS fvu=$FVU"
    echo -e "${TIMESTAMP}\t${COMPOSITE}\t${CLINC}\t${SCIFACT}\t${MMCS}\t${FVU}\t${ACTIVE}\tok\texperiment_${EXP_NUM}" >> "$RESULTS_TSV"

    # Brief pause between experiments
    sleep 2
done

echo ""
echo "[Agent $AGENT_ID] Loop complete. Ran $EXP_NUM experiments."
echo "[Agent $AGENT_ID] Results in $RESULTS_TSV"
