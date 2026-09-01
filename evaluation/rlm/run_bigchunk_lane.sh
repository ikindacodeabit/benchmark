#!/bin/bash
# Arm 4d -- RLM whose sub-calls are big enough for the KV press to do anything.
#
# WHY THIS ARM EXISTS. Arms 4a/4b/4c all reported sub_pressed_call_fraction 0.0:
# the press was installed and never engaged. The cause was the slice size, not the
# press. Across the finished LOFT-128k and LOFT-1m campaigns the root sent a mean
# of 1023 characters per llm_query against a 32000-character cap -- 96% of slices
# under 2000 chars, not one call ever reaching the cap -- because the prompt's
# worked example demonstrates a 700-char keyhole and the root copies it. Raising
# the cap does not help (32000 -> 131072 chars moved the realized slice from 276
# to 384 tokens): a cap is a ceiling the root MAY use, not a floor that makes it
# read. --min-subcall-chars is that floor.
#
# WHY 65536 AND NOT SOMETHING SMALLER. Two thresholds, and only the second bites.
# --press-min-tokens (1024) decides whether the press RUNS. The KV token budget
# decides whether it EVICTS anything, because each call's ratio is
# 1 - min(ctx, budget)/ctx. Qwen3-4B costs 144 KiB/token, so a 1 GB budget is
# ~6781 tokens: a 16384-char (~4k token) slice clears press_min_tokens and still
# compresses by EXACTLY ZERO. 65536 chars is ~16.3k tokens -> measured ratio
# 0.585 at 1 GB, and comfortably inside --sub-max-context-tokens (34000).
#
# WHY A LANE RATHER THAN run_infolab.sh auto. That path gates arm-4 colocation at
# MIN_FREE_MIB=38000 and starts its own vLLM. On a busy shared infolab host the
# free cards sit just under that gate while a perfectly good root server is
# already running. This reuses the server and puts only the sub model on $GPU.
#
# Usage (from the repo root, under tmux -- an ssh drop kills the run):
#   DATASET=nq LENGTH=1m LIMIT=110 GPU=4 PORT=8000 bash $0
#   # sanity check first, 3 examples:
#   DATASET=nq LENGTH=1m LIMIT=3 GPU=4 SMOKE=1 bash $0
set -euo pipefail

DATASET="${DATASET:-nq}"
LENGTH="${LENGTH:-1m}"
LIMIT="${LIMIT:-110}"
FLOOR="${FLOOR:-65536}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
BUDGET="${BUDGET:-1}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_SUB_CALLS="${MAX_SUB_CALLS:-12}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# The arm-4 driver needs kvpress, which needs transformers>=4.56 for the Cache API
# -- while the vLLM server needs 4.51.3 (0.8.5 calls all_special_tokens_extended,
# removed in 5.x). The two pins coexist only because they are separate PROCESSES:
# this driver only speaks HTTP to the server.
KVPY="${KVZIP_PYTHON:-$REPO_ROOT/.venv-kvpress/bin/python}"
RESULTS="${RESULTS:-evaluation/results/loft${LENGTH}_bigchunk$([ -n "${SMOKE:-}" ] && echo _smoke || true)}"

if [ ! -x "$KVPY" ]; then
    echo "ERROR: no kvpress venv python at $KVPY (set KVZIP_PYTHON)" >&2
    exit 1
fi
if ! curl -sf -m 10 "http://localhost:$PORT/v1/models" >/dev/null; then
    echo "ERROR: no vLLM server answering on localhost:$PORT." >&2
    echo "  Start one, or point PORT at an existing server." >&2
    exit 1
fi

cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-/mnt/nas/$USER/hf_cache}"
# The compute path needs nothing from the Hub once the subsets are staged, and the
# shared infolab IP is rate-limited hard enough to stall a run.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== $(date '+%F %T') :: arm 4d ${DATASET}_${LENGTH} floor=${FLOOR} budget=${BUDGET}GB gpu=${GPU} ==="
CUDA_VISIBLE_DEVICES="$GPU" "$KVPY" -m evaluation.rlm.run_benchmark \
    --dataset loft --data-dir "${DATASET}_${LENGTH}" --limit "$LIMIT" \
    --base-url "http://localhost:$PORT/v1" \
    --root-model "$MODEL" --sub-model "$MODEL" \
    --max-steps 50 --exec-timeout 60 \
    --out "$RESULTS" \
    --mode rlm --scratchpad --max-notes-tokens 1024 \
    --sub-backend kvzip --press kvzip --sub-max-tokens 512 \
    --memory-budget "$BUDGET" --memory-budget-unit GB \
    --max-subcall-chars 131072 --min-subcall-chars "$FLOOR" \
    --max-sub-calls "$MAX_SUB_CALLS" --run-timeout "$RUN_TIMEOUT"
echo "=== $(date '+%F %T') :: arm 4d ${DATASET}_${LENGTH} done ==="
echo "Check: runtime.sub_pressed_call_fraction (must be > 0; it was 0.0 for arms 4a/4b/4c),"
echo "       runtime.average_sub_compression_ratio, runtime.average_sub_payload_chars."
