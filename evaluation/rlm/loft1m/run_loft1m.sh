#!/bin/bash
# LOFT-1m: the regime where the RLM hypothesis is actually testable.
#
# WHY 1m AND NOT 128k. Qwen3-4B-2507 has a 262144-token window and a LOFT-128k
# document is ~119k tokens, so the vanilla baseline sees ALL of it:
# context_retained is 1.0 and RLM pays recursion overhead for a constraint that
# does not exist. That is why vanilla won every 128k subset -- a fact about the
# regime, not about RLM. A LOFT-1m document is ~931k tokens (~3.7M characters),
# of which the baseline reads 124,851 -- context_retained 0.134. The other 87%
# is unreachable without recursion.
#
# Usage (from the repository root, under tmux -- an ssh drop kills the servers):
#
#   # Lanes for N subsets, one fresh server per usable card:
#   SERVERS=2 DATASETS="hotpotqa musique qampari quest" bash $0 auto
#
#   # One subset against a server that is ALREADY running on $PORT. Reusing a
#   # loaded server is how a host with only two free cards still runs three lanes.
#   DATASET=nq PORT=8000 bash $0 lane
#
#   # 10-example vanilla smoke: reads context_retained, the gate on whether the
#   # grid is worth running at all. If it comes back 1.0 the caps did not bind
#   # and the grid would only re-prove the 128k result.
#   DATASET=nq bash $0 smoke
#
# THE TWO CAPS THAT MATTER, and why they are not run_cells.sh's defaults:
#   VANILLA_CHAR_LIMIT=4000000 -- the 700k default binds BEFORE the token cap on
#     a 3.7M-character document, so context_retained would report the truncation
#     ratio (~0.19) rather than the fraction the model actually read (0.134).
#     Above the document size the char cap goes inert and the TOKEN cap governs.
#   RUN_TIMEOUT=1800 -- 900s is tight for a search over 3.7M characters.
set -euo pipefail

case "${1:-}" in
auto | lane | smoke) ;;
*)
    echo "usage: $0 {auto|lane|smoke}" >&2
    exit 2
    ;;
esac

# Same scratch layout as run_infolab.sh: $HOME is a quota-capped NFS mount on the
# infolab hosts and a large write to it has taken the home directory down
# host-wide before.
RLM_SCRATCH="${RLM_SCRATCH:-/mnt/nas/$USER}"
if [ ! -d "$RLM_SCRATCH" ]; then
    echo "ERROR: RLM_SCRATCH=$RLM_SCRATCH does not exist." >&2
    echo "  Set it to a big writable path before running." >&2
    exit 1
fi
export RLM_SCRATCH
export HF_HOME="${HF_HOME:-$RLM_SCRATCH/hf_cache}"
# The compute path needs nothing from the Hub once download_data.sh has staged
# the subsets, and the shared infolab IP is rate-limited hard enough to stall a
# run. Offline turns a 429 stall into an immediate, legible error.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
VENV="${VENV:-$REPO_ROOT/.venv}"
LENGTH="${LENGTH:-1m}"
LIMIT="${LIMIT:-110}"
RESULTS="${RESULTS:-evaluation/results/loft1m}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
VANILLA_CHAR_LIMIT="${VANILLA_CHAR_LIMIT:-4000000}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"
export VANILLA_CHAR_LIMIT RUN_TIMEOUT

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: no venv at $VENV; set VENV=<path to your venv>" >&2
    exit 1
fi

case "$1" in
auto)
    # Delegates to run_infolab.sh, which owns GPU selection, memory sizing and
    # readiness -- this wrapper only supplies the 1m caps.
    PORT="${PORT:-8001}" SERVERS="${SERVERS:-2}" LENGTH="$LENGTH" LIMIT="$LIMIT" \
        DATASETS="${DATASETS:-hotpotqa musique qampari quest}" \
        RESULTS="$RESULTS" VENV="$VENV" \
        bash evaluation/rlm/loft128k/run_infolab.sh auto
    ;;

lane)
    # run_cells.sh's arms 1-3 call a bare `python`, so the venv must be active:
    # run_infolab.sh sources it before delegating, and this path bypasses that.
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    DATASET="${DATASET:?set DATASET=<nq|hotpotqa|musique|qampari|quest>}" \
        LENGTH="$LENGTH" PORT="${PORT:-8000}" LIMIT="$LIMIT" \
        RESULTS="$RESULTS" MODEL="$MODEL" VENV="$VENV" \
        bash evaluation/rlm/loft128k/run_cells.sh
    ;;

smoke)
    # Vanilla only, and deliberately: the gate is context_retained on the vanilla
    # rows, which the RLM arms cannot affect, so there is no reason to spend an
    # RLM arm's wall-clock before deciding whether to run the grid at all.
    # Its OWN results dir -- `limit` is resume-critical but is NOT part of the run
    # directory slug, so a LIMIT=10 smoke and the LIMIT=110 grid would collide in
    # one directory and the grid would trip the resume guard on the smoke's rows.
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m evaluation.rlm.run_benchmark \
        --dataset loft \
        --data-dir "${DATASET:?set DATASET=<subset>}_${LENGTH}" \
        --limit "${SMOKE_LIMIT:-10}" \
        --base-url "http://localhost:${PORT:-8000}/v1" \
        --root-model "$MODEL" \
        --sub-model "$MODEL" \
        --max-steps 50 \
        --vanilla-char-limit "$VANILLA_CHAR_LIMIT" \
        --vanilla-max-prompt-tokens 134000 \
        --exec-timeout 60 \
        --run-timeout "$RUN_TIMEOUT" \
        --max-sub-calls 40 \
        --out "${SMOKE_RESULTS:-evaluation/results/loft1m_smoke}" \
        --mode vanilla
    ;;
esac
