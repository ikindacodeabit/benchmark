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
# The kvpress arm gets its OWN lock: its run dirs are disjoint from arms 1-3
# (the slug carries press/ratio/chunk suffixes), so the checkpoint-merge hazard
# is only between two runs of the SAME lane family.
LOCK="$LOGS/.lock.$DATASET${KVPRESS_ARMS:+.kvpress}"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another run of '$DATASET' already holds $LOCK; exiting" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
exec > >(tee -a "$LOGS/$DATASET.$STAMP${KVPRESS_ARMS:+.kvpress}.out") 2>&1

echo "=== $(date '+%F %T') :: loft ${DATASET}_${LENGTH} :: $MODEL :: port $PORT ==="

# --- wait for the server ------------------------------------------------------
for _ in $(seq 1 "${READY_TRIES:-90}"); do
    if curl -sf -o /dev/null --max-time 5 "http://localhost:$PORT/v1/models"; then
        echo "vLLM ready on port $PORT"
        break
    fi
    sleep 10
done
if ! curl -sf -o /dev/null --max-time 5 "http://localhost:$PORT/v1/models"; then
    echo "ERROR: no vLLM on port $PORT after $(( ${READY_TRIES:-90} * 10 / 60 )) minutes" >&2
    exit 1
fi

# --- arm 4: rlm + scratchpad + KV-compressed sub-calls ------------------------
# KVPRESS_ARMS=1 runs ONLY these lanes and then exits. Deliberate: this mode is
# served by a SMALL root server (see run_infolab.sh colocation sizing), and a
# vanilla arm pointed at it would shrink-retry against the reduced window and
# merge truncated rows into the existing vanilla checkpoints. The driver runs
# from the kvpress venv; the sub model is hosted in-process on RLM_SUB_GPU.
if [ -n "${KVPRESS_ARMS:-}" ]; then
    KVPY="${KVPRESS_PYTHON:-.venv-kvpress/bin/python}"
    if [ ! -x "$KVPY" ]; then
        echo "ERROR: no kvpress venv python at $KVPY; run run_infolab.sh setup first" >&2
        exit 1
    fi
    KV_BASE=(
        --dataset loft
        --data-dir "${DATASET}_${LENGTH}"
        --limit "$LIMIT"
        --base-url "http://localhost:$PORT/v1"
        --root-model "$MODEL"
        --sub-model "$MODEL"
        --rpm 600
        --max-steps 50
        --exec-timeout 60
        --out "$RESULTS"
        --mode rlm
        --scratchpad
        --max-notes-tokens "$MAX_NOTES_TOKENS"
        --sub-backend kvpress
        --press "${KV_PRESS:-kvzip}"
        --sub-max-tokens "${KV_SUB_MAX_TOKENS:-512}"
    )
    FAILED=0
    for ratio in ${KV_RATIOS:-0.5 0.75}; do
        for arm in ${KV_ARMS:-4a 4b}; do
            case "$arm" in
            # 4a keeps the 32k-char chunking of arms 2/3 (isolates the press
            # effect); 4b raises it to ~32k TOKENS (fewer, bigger reads -- the
            # regime compression is supposed to enable). KVzip scores in 2-3
            # extra prefill passes, so 4b calls run ~a minute each: fewer calls,
            # longer per-example budget.
            4a) EXTRA=(--max-subcall-chars 32000 --max-sub-calls 40 --run-timeout "${KV_RUN_TIMEOUT_4A:-2400}") ;;
            4b) EXTRA=(--max-subcall-chars 131072 --max-sub-calls 16 --run-timeout "${KV_RUN_TIMEOUT_4B:-3600}") ;;
            *)
                echo "ERROR: unknown KV_ARMS entry '$arm' (use 4a and/or 4b)" >&2
                exit 2
                ;;
            esac
            echo "--- arm $arm: rlm+scratchpad+${KV_PRESS:-kvzip} ratio $ratio ---"
            CUDA_VISIBLE_DEVICES="${RLM_SUB_GPU:-0}" "$KVPY" -m evaluation.rlm.run_benchmark \
                "${KV_BASE[@]}" --compression-ratio "$ratio" "${EXTRA[@]}" ||
                {
                    echo "WARN: ${DATASET} arm $arm ratio $ratio failed"
                    FAILED=1
                }
        done
    done
    echo "=== $(date '+%F %T') :: ${DATASET}_${LENGTH} kvpress arms done (failed=$FAILED) ==="
    exit "$FAILED"
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
