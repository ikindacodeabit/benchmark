#!/bin/bash
# Fixed-chunk x logical-KV-retention grid for the RLM.
#
# Reuses an already-running local OpenAI-compatible root server. KVzipPress masks
# keys but does NOT free their memory: GPU fit depends on N=B*F, not on B. The
# token-budget rows are therefore simulated retention budgets, not measured
# memory savings.
set -euo pipefail

DATASETS="${DATASETS:-musique hotpotqa 2wikimqa narrativeqa qasper triviaqa}"
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
LIMIT="${LIMIT:-50}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RESULTS="${RESULTS:-evaluation/results/fixedgrid}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
KVPY="${KVZIP_PYTHON:-$REPO_ROOT/.venv-kvpress/bin/python}"
LOCK_DIR="${LOCK_DIR:-$RESULTS/.locks}"

if [ ! -x "$KVPY" ]; then
    echo "ERROR: no kvpress Python at $KVPY (set KVZIP_PYTHON)" >&2
    exit 1
fi
if ! curl -sf -m 10 "http://localhost:$PORT/v1/models" | grep -qF "\"$MODEL\""; then
    echo "ERROR: localhost:$PORT is not serving $MODEL" >&2
    exit 1
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
            calls=$(((131072 + n - 1) / n + 1))
            if [ "$calls" -gt 64 ]; then calls=64; fi
            min_free=$(awk -v n="$n" 'BEGIN{x=n*147456*1.2/(2^30); m=int(x); if(x>m)m++; m+=12; if(m<14)m=14; print m}')
            press=kvzip
            if [ "$factor" -eq 1 ]; then press=no_press; fi

            run_name="longbench128k__${subset}__${model_slug}__rlm__scratchpad__kvzip-${press}${budget}tokens__autosubx${factor}__fixed"
            if [ -f "$RESULTS/$run_name/metrics.json" ]; then
                echo "skip complete: $subset B=$budget F=$factor"
                continue
            fi

            # Cell-specific locking permits budget/factor parallelism while
            # preventing two workers from interleaving one checkpoint.jsonl.
            lock="$LOCK_DIR/${subset}.${budget}.${factor}.lock"
            exec 9>"$lock"
            if ! flock -n 9; then
                echo "skip locked: $subset B=$budget F=$factor"
                continue
            fi

            echo "=== $subset :: B=$budget tokens :: F=${factor}x :: N=$n :: calls=$calls :: minfree=${min_free}GiB ==="
            if ! CUDA_VISIBLE_DEVICES="$GPU" "$KVPY" -m evaluation.rlm.run_benchmark \
                --dataset longbench128k --data-dir "$subset" --limit "$LIMIT" \
                --base-url "http://localhost:$PORT/v1" \
                --root-model "$MODEL" --sub-model "$MODEL" \
                --mode rlm --scratchpad --max-steps 50 --exec-timeout 60 --run-timeout 3600 \
                --sub-backend kvzip --press "$press" \
                --memory-budget "$budget" --memory-budget-unit tokens \
                --max-subcall-chars auto --compression-factor "$factor" --fixed-chunk \
                --max-sub-calls "$calls" --sub-max-tokens 128 --sub-min-free-gib "$min_free" \
                --subcall-min-tokens "$MIN_TOKENS" \
                --out "$RESULTS"; then
                echo "!! CELL FAILED: $subset B=$budget F=$factor N=$n -- continuing" >&2
                echo "$subset $budget $factor $n" >> "$RESULTS/failed_cells.txt"
            fi
            flock -u 9
        done
    done
done
