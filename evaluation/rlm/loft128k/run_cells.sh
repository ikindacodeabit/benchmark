#!/bin/bash
# Host-agnostic driver for one LOFT dataset: runs the three arms against an
# ALREADY-RUNNING vLLM server. The SLURM script and the bee script both call this,
# so the flags that define the experiment live in exactly one place.
#
# Usage (from repository root, with a server on $PORT):
#   DATASET=nq PORT=8000 RESULTS=evaluation/results/loft128k \
#     bash evaluation/rlm/loft128k/run_cells.sh
#
# Arms: vanilla, rlm, rlm+scratchpad. `--mode both` covers the first two in one
# pass so the vanilla arm is not recomputed for the scratchpad comparison.
set -euo pipefail

DATASET="${DATASET:?set DATASET=<nq|hotpotqa|musique|qampari|quest>}"
LENGTH="${LENGTH:-128k}"
PORT="${PORT:-8000}"
LIMIT="${LIMIT:-110}"
RESULTS="${RESULTS:-evaluation/results/loft128k}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_NOTES_TOKENS="${MAX_NOTES_TOKENS:-1024}"
LOGS="${LOGS:-$RESULTS/logs}"

# client.py requires this even against a local server; the value is ignored.
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-local-dummy}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$RESULTS" "$LOGS"

# --- duplicate-run guard ------------------------------------------------------
# run_benchmark reads its checkpoint ONCE per cell, so a second concurrent run of
# the same dataset cannot see the first's in-flight rows and would redo all the
# remaining work, then interleave writes into the same checkpoint.jsonl.
LOCK="$LOGS/.lock.$DATASET"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another run of '$DATASET' already holds $LOCK; exiting" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
exec > >(tee -a "$LOGS/$DATASET.$STAMP.out") 2>&1

echo "=== $(date '+%F %T') :: loft ${DATASET}_${LENGTH} :: $MODEL :: port $PORT ==="

# --- wait for the server ------------------------------------------------------
for _ in $(seq 1 90); do
    if curl -sf -o /dev/null --max-time 5 "http://localhost:$PORT/v1/models"; then
        echo "vLLM ready on port $PORT"
        break
    fi
    sleep 10
done
if ! curl -sf -o /dev/null --max-time 5 "http://localhost:$PORT/v1/models"; then
    echo "ERROR: no vLLM on port $PORT after 15 minutes" >&2
    exit 1
fi

# --- shared flags -------------------------------------------------------------
# --vanilla-char-limit: a 131,072-token LOFT context is roughly 520k characters.
# The 400k default would silently cut the baseline to ~77% of the document and
# hand the RLM arm a win it did not earn. Bound the prompt by TOKENS instead.
#
# NOTE: no --no-think and no --reasoning-parser anywhere in this pipeline.
# Qwen3-*-Instruct-2507 is a non-thinking model; it emits no <think> blocks, and
# the enable_thinking chat-template kwarg does not apply to its template.
COMMON=(
    --dataset loft
    --data-dir "${DATASET}_${LENGTH}"
    --limit "$LIMIT"
    --base-url "http://localhost:$PORT/v1"
    --root-model "$MODEL"
    --sub-model "$MODEL"
    --rpm 600
    --max-steps 50
    --vanilla-char-limit 700000
    --vanilla-max-prompt-tokens 134000
    --exec-timeout 60
    --run-timeout 900
    --max-sub-calls 40
    --out "$RESULTS"
)

echo "--- arms 1+2: vanilla and rlm ---"
python -m evaluation.rlm.run_benchmark "${COMMON[@]}" --mode both \
    || echo "WARN: ${DATASET} vanilla/rlm failed; continuing to the scratchpad arm"

echo "--- arm 3: rlm + scratchpad ---"
python -m evaluation.rlm.run_benchmark "${COMMON[@]}" --mode rlm \
    --scratchpad --max-notes-tokens "$MAX_NOTES_TOKENS" \
    || echo "WARN: ${DATASET} rlm+scratchpad failed"

echo "=== $(date '+%F %T') :: ${DATASET}_${LENGTH} done ==="
