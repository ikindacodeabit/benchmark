#!/bin/bash
# Fixed-chunk x logical-KV-retention grid for the RLM.
#
# Reuses an already-running local OpenAI-compatible root server. KVzipPress masks
# keys but does NOT free their memory: GPU fit depends on N=B*F, not on B. The
# token-budget rows are therefore simulated retention budgets, not measured
# memory savings.
set -euo pipefail

DATASETS="${DATASETS:-musique hotpotqa 2wikimqa narrativeqa qasper triviaqa}"
# The benchmark the subsets belong to. longbench128k is a locally built pack of
# 131072-token rows; loft is the Hub-mirrored RAG set, whose subsets carry a
# length suffix (`nq_1m`) and whose 1m rows are ~7x longer. Those two facts are
# the only things that differ per benchmark, so they are parameters rather than
# a second copy of this script.
DATASET_NAME="${DATASET_NAME:-longbench128k}"
TASK_SUFFIX="${TASK_SUFFIX:-}"
# Drives --max-sub-calls: a lane must be able to sweep the WHOLE document, and
# the count that achieves that is ceil(DOC_TOKENS / N) + 1. Left at the
# longbench128k row size, a LOFT-1m lane would budget a 14% sweep and measure
# the cap rather than the cell.
DOC_TOKENS="${DOC_TOKENS:-131072}"
# Ceiling on that derivation. Each sub-call is a full prefill, so this bounds a
# cell's wall clock; small-N cells at 1m would otherwise ask for hundreds.
MAX_CALLS_CAP="${MAX_CALLS_CAP:-64}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
# LOFT concatenates dev (10 rows) then test (100), so a limited lane that does
# not say which split it wants silently evaluates dev-heavy rows. Empty keeps the
# harness default (`all`), which is what every longbench128k run used.
SPLIT="${SPLIT:-}"
BUDGETS="${BUDGETS:-1024 2048 4096 8192}"
FACTORS="${FACTORS:-1 2 4 8 16}"
# Qwen3-4B-Instruct-2507: max_position_embeddings, minus the sizing reserve.
# A cell with N above this cannot be served at all -- --fixed-chunk refuses
# rather than silently shrinking, so it is skipped here instead.
SUB_WINDOW="${SUB_WINDOW:-262144}"
RESERVE_TOKENS="${RESERVE_TOKENS:-1024}"
# Auto-sizing refuses below this. The 1024 default predates budget-derived
# press_min_tokens and would block every B<1024 cell; grid lanes set it to
# the smallest N they intend to run.
MIN_TOKENS="${MIN_TOKENS:-1024}"
# Per-cell --sub-min-free-gib is derived below as 2.0x the uncompressed KV plus a
# flat 12 GiB for weights and slack. The 2.0x is measured, not assumed: KVzip's
# reconstruction-scoring pass roughly doubles peak memory over the cache itself
# (a LOFT-1m cell at N=108504 held 38.8 GiB against a 16.0 GiB raw KV).
#
# This override exists for a card whose free memory lands just under the derived
# figure. Use it to shave slack, NEVER to get under the 2.0x: a cell that does
# not fit does not fail loudly. The sub model returns its out-of-memory string as
# an ordinary answer, the root burns its call budget retrying, llm_query is
# withdrawn, and the cell COMPLETES with a score describing an RLM that had no
# sub-model. Lowering this to 20 on a 20 GiB card cost a LOFT-1m lane exactly
# that. It applies to every cell in the lane, so scope such a lane narrowly.
MIN_FREE_GIB="${MIN_FREE_GIB:-}"
LIMIT="${LIMIT:-50}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
# The host serving the ROOT model. Normally the machine running the lane, but the
# root is a plain OpenAI-compatible endpoint doing light work (peak context ~2.4k
# tokens), so a host that cannot serve can borrow one over the LAN. That is not a
# nicety on ant: its Blackwell cards are sm_120 and the serving venv's torch has
# no kernels for them, so vllm cannot start there at all, while the sub model
# runs fine from the cu128 venv.
ROOT_HOST="${ROOT_HOST:-localhost}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RESULTS="${RESULTS:-evaluation/results/fixedgrid}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
KVPY="${KVZIP_PYTHON:-$REPO_ROOT/.venv-kvpress/bin/python}"
LOCK_DIR="${LOCK_DIR:-$RESULTS/.locks}"

if [ ! -x "$KVPY" ]; then
    echo "ERROR: no kvpress Python at $KVPY (set KVZIP_PYTHON)" >&2
    exit 1
fi
if ! curl -sf -m 10 "http://$ROOT_HOST:$PORT/v1/models" | grep -qF "\"$MODEL\""; then
    echo "ERROR: $ROOT_HOST:$PORT is not serving $MODEL" >&2
    exit 1
fi
# longbench128k is read from a locally built tree; the loader takes its root from
# RLM_DATA_DIR and silently falls back to ~/rlm_data, which on a shared-NAS host
# is the wrong tree and does not exist. That surfaced only after the model had
# loaded, one cell at a time; check it once, up front, for every subset the lane
# is about to run. LOFT comes from the HF cache instead and has no such tree, so
# the check is scoped to the benchmark that needs it.
if [ "$DATASET_NAME" = "longbench128k" ]; then
    : "${RLM_DATA_DIR:?set RLM_DATA_DIR to the tree holding longbench128k/<subset>/data.parquet}"
    for subset in $DATASETS; do
        pack="$RLM_DATA_DIR/longbench128k/$subset/data.parquet"
        if [ ! -f "$pack" ]; then
            echo "ERROR: missing data pack $pack" >&2
            echo "       build it with: python -m evaluation.benchmarks.longbench128k.build_dataset" >&2
            exit 1
        fi
    done
fi

mkdir -p "$RESULTS" "$LOCK_DIR"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mnt/nas/$USER/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

model_slug="${MODEL//\//_}"
for subset in $DATASETS; do
    for budget in $BUDGETS; do
        for factor in $FACTORS; do
            if ! [[ "$budget" =~ ^[0-9]+$ && "$factor" =~ ^[0-9]+$ ]]; then
                echo "ERROR: BUDGETS and FACTORS must contain positive integers" >&2
                exit 2
            fi
            n=$((budget * factor))
            window_cap=$((SUB_WINDOW - RESERVE_TOKENS))
            if [ "$n" -gt "$window_cap" ]; then
                echo "skip infeasible: $subset B=$budget F=$factor N=$n > window cap $window_cap"
                continue
            fi
            if [ "$n" -lt "$MIN_TOKENS" ]; then
                echo "skip below floor: $subset B=$budget F=$factor N=$n < MIN_TOKENS $MIN_TOKENS"
                continue
            fi
            calls=$(((DOC_TOKENS + n - 1) / n + 1))
            if [ "$calls" -gt "$MAX_CALLS_CAP" ]; then calls="$MAX_CALLS_CAP"; fi
            min_free=$(awk -v n="$n" 'BEGIN{x=n*147456*2.0/(2^30); m=int(x); if(x>m)m++; m+=12; if(m<14)m=14; print m}')
            if [ -n "$MIN_FREE_GIB" ]; then min_free="$MIN_FREE_GIB"; fi
            press=kvzip
            if [ "$factor" -eq 1 ]; then press=no_press; fi

            task="${subset}${TASK_SUFFIX}"
            # Mirrors build_run_dir_components: the harness names the directory
            # itself, and this only needs to predict it to skip a finished cell.
            # The split component sits between the mode and `scratchpad` there,
            # and is omitted for `all` -- get the order wrong and the skip never
            # matches, so every finished cell reloads the model to do nothing.
            split_slug=""
            if [ -n "$SPLIT" ] && [ "$SPLIT" != "all" ]; then split_slug="__split-${SPLIT}"; fi
            run_name="${DATASET_NAME}__${task}__${model_slug}__rlm${split_slug}__scratchpad__kvzip-${press}${budget}tokens__autosubx${factor}__fixed"
            if [ -f "$RESULTS/$run_name/metrics.json" ]; then
                echo "skip complete: $subset B=$budget F=$factor"
                continue
            fi

            # Cell-specific locking permits budget/factor parallelism while
            # preventing two workers from interleaving one checkpoint.jsonl.
            # Keyed on the TASK, not the subset: `nq_1m` and `nq_128k` are
            # different cells that would otherwise share one lock.
            lock="$LOCK_DIR/${task}.${budget}.${factor}.lock"
            exec 9>"$lock"
            if ! flock -n 9; then
                echo "skip locked: $subset B=$budget F=$factor"
                continue
            fi

            split_args=()
            if [ -n "$SPLIT" ]; then split_args=(--split "$SPLIT"); fi

            echo "=== $task :: B=$budget tokens :: F=${factor}x :: N=$n :: calls=$calls :: minfree=${min_free}GiB ==="
            if ! CUDA_VISIBLE_DEVICES="$GPU" "$KVPY" -m evaluation.rlm.run_benchmark \
                --dataset "$DATASET_NAME" --data-dir "$task" --limit "$LIMIT" "${split_args[@]}" \
                --base-url "http://$ROOT_HOST:$PORT/v1" \
                --root-model "$MODEL" --sub-model "$MODEL" \
                --mode rlm --scratchpad --max-steps 50 --exec-timeout 60 --run-timeout "$RUN_TIMEOUT" \
                --sub-backend kvzip --press "$press" \
                --memory-budget "$budget" --memory-budget-unit tokens \
                --max-subcall-chars auto --compression-factor "$factor" --fixed-chunk \
                --max-sub-calls "$calls" --sub-max-tokens 128 --sub-min-free-gib "$min_free" \
                --subcall-min-tokens "$MIN_TOKENS" \
                --out "$RESULTS"; then
                echo "!! CELL FAILED: $task B=$budget F=$factor N=$n -- continuing" >&2
                echo "$task $budget $factor $n" >> "$RESULTS/failed_cells.txt"
            fi
            flock -u 9
        done
    done
done
