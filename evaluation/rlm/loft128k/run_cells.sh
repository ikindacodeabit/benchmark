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

# client.py no longer needs NVIDIA_API_KEY for a local base_url; only the
# hosted catalog requires one.
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Caches must never land on $HOME: it is a small, invisible NFS quota on the
# infolab hosts, and the kvzip arm pulls ~8 GB of sub-model weights. run_infolab.sh
# redirects them, but the usage note above advertises driving this script directly
# against an already-running server, and that path bypassed the redirect -- on bee
# 2026-08-18 the sub model downloaded into $HOME and died with EDQUOT mid-symlink,
# taking the whole 4a lane with it. Repeat the redirect here so both entry points
# are safe, and fail loudly rather than filling a quota-capped home.
RLM_SCRATCH="${RLM_SCRATCH:-/mnt/nas/$USER}"
export HF_HOME="${HF_HOME:-$RLM_SCRATCH/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RLM_SCRATCH/cache}"
case "$HF_HOME" in
"$HOME" | "$HOME"/*)
    echo "ERROR: HF_HOME=$HF_HOME is under \$HOME, which is quota-capped here." >&2
    echo "  Set HF_HOME (or RLM_SCRATCH) to a big writable path before running." >&2
    exit 1
    ;;
esac
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

mkdir -p "$RESULTS" "$LOGS"

# --- duplicate-run guard ------------------------------------------------------
# run_benchmark reads its checkpoint ONCE per cell, so a second concurrent run of
# the same dataset cannot see the first's in-flight rows and would redo all the
# remaining work, then interleave writes into the same checkpoint.jsonl.
# The kvzip arm gets its OWN lock: its run dirs are disjoint from arms 1-3
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
# The grep for $MODEL is load-bearing: on a shared host another user's server can
# hold the port, and a bare 200-check would accept it -- every request then 404s
# with "model does not exist" while readiness claims all is well.
serves_our_model() {
    curl -sf --max-time 5 "http://localhost:$PORT/v1/models" | grep -qF "\"$MODEL\""
}
for _ in $(seq 1 "${READY_TRIES:-90}"); do
    if serves_our_model; then
        echo "vLLM serving $MODEL on port $PORT"
        break
    fi
    sleep 10
done
if ! serves_our_model; then
    echo "ERROR: no server for $MODEL on port $PORT after $(( ${READY_TRIES:-90} * 10 / 60 )) minutes." >&2
    echo "  If /v1/models answers but with a different model, the port belongs to another user." >&2
    exit 1
fi

# --- arm 4: rlm + scratchpad + KV-compressed sub-calls ------------------------
# KVPRESS_ARMS=1 runs ONLY these lanes and then exits. Deliberate: this mode is
# served by a SMALL root server (see run_infolab.sh colocation sizing), and a
# vanilla arm pointed at it would shrink-retry against the reduced window and
# merge truncated rows into the existing vanilla checkpoints. The driver runs
# from the MAIN venv (KVzip pins transformers==4.51.3, same as the vLLM server,
# so no separate venv is needed); the sub model is hosted in-process on
# RLM_SUB_GPU.
if [ -n "${KVPRESS_ARMS:-}" ]; then
    KVPY="${KVZIP_PYTHON:-.venv/bin/python}"
    if [ ! -x "$KVPY" ]; then
        echo "ERROR: no venv python at $KVPY; run run_infolab.sh setup first (or set KVZIP_PYTHON)" >&2
        exit 1
    fi
    # No KVzip checkout needed any more: the sub-backend uses kvpress's own
    # KVzipPress through KVPressTextGenerationPipeline, not snu-mllab/KVzip.
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
        --sub-backend kvzip
        --press "${KV_PRESS:-kvzip}"
        --sub-max-tokens "${KV_SUB_MAX_TOKENS:-512}"
    )
    FAILED=0
    # KV budgets, not compression ratios: the press takes a memory budget and
    # derives each call's ratio from the slice it actually receives. These are
    # matrix_constants.EXTENDED_KV_BUDGETS values so the arm-4 numbers sit on the
    # same axis as the LOFT/RULER matrix runs.
    for budget in ${KV_BUDGETS:-0.512 1}; do
        for arm in ${KV_ARMS:-4a 4b}; do
            case "$arm" in
            # 4a keeps the 32k-char chunking of arms 2/3 (isolates the press
            # effect); 4b raises it to ~32k TOKENS (fewer, bigger reads -- the
            # regime compression is supposed to enable). KVzip scores in 2-3
            # extra prefill passes, so 4b calls run ~a minute each: fewer calls,
            # longer per-example budget.
            # 4c derives the chunk from the budget instead of hand-picking it, so
            # the realized compression ratio is held constant across budgets
            # rather than drifting with whatever size was chosen. Kept as a THIRD
            # arm rather than retargeting 4a/4b, whose results are mid-campaign.
            4a) EXTRA=(--max-subcall-chars 32000 --max-sub-calls 40 --run-timeout "${KV_RUN_TIMEOUT_4A:-2400}") ;;
            4b) EXTRA=(--max-subcall-chars 131072 --max-sub-calls 16 --run-timeout "${KV_RUN_TIMEOUT_4B:-3600}") ;;
            4c) EXTRA=(
                --max-subcall-chars auto
                --target-compression-ratio "${KV_TARGET_RATIO:-0.75}"
                --max-sub-calls 16
                --run-timeout "${KV_RUN_TIMEOUT_4C:-3600}"
            ) ;;
            *)
                echo "ERROR: unknown KV_ARMS entry '$arm' (use 4a, 4b and/or 4c)" >&2
                exit 2
                ;;
            esac
            echo "--- arm $arm: rlm+scratchpad+${KV_PRESS:-kvzip} budget ${budget}GB ---"
            CUDA_VISIBLE_DEVICES="${RLM_SUB_GPU:-0}" "$KVPY" -m evaluation.rlm.run_benchmark \
                "${KV_BASE[@]}" --memory-budget "$budget" --memory-budget-unit GB "${EXTRA[@]}" ||
                {
                    echo "WARN: ${DATASET} arm $arm budget ${budget}GB failed"
                    FAILED=1
                }
        done
    done
    echo "=== $(date '+%F %T') :: ${DATASET}_${LENGTH} kvzip arms done (failed=$FAILED) ==="
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
