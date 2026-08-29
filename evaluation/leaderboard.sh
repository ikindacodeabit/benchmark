#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script to run the leaderboard evaluation on 4 GPUs. Run from evaluation/.
# Override any setting from the environment, e.g. MODEL=meta-llama/... ./leaderboard.sh
set -euo pipefail

DATASET="${DATASET:-ruler}"
DATA_DIR="${DATA_DIR:-4096}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PYTHON="${PYTHON:-python}"

# Anchored on this script, not the caller's cwd: the old relative path silently
# wrote somewhere else when invoked from anywhere but evaluation/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../benchmark_artifacts/results/leaderboard}"

# kvzap thresholds are model-specific. These used to be hardcoded to the
# Qwen3-8B values with only a comment saying Llama needed different ones, so
# changing MODEL silently produced wrong thresholds.
case "${MODEL}" in
    *Llama-3.1-8B*) KVZAP_THRESHOLDS=(-6 -7 -8 -9) ;;
    *Qwen3-8B*)     KVZAP_THRESHOLDS=(-3 -4 -5 -6) ;;
    *)
        if [ -z "${KVZAP_THRESHOLDS:-}" ]; then
            echo "Error: no kvzap thresholds known for MODEL=${MODEL}." >&2
            echo "Set KVZAP_THRESHOLDS, e.g. KVZAP_THRESHOLDS='-3 -4 -5 -6'" >&2
            exit 1
        fi
        read -r -a KVZAP_THRESHOLDS <<< "${KVZAP_THRESHOLDS}"
        ;;
esac

# Run one configuration per GPU and fail loudly if any of them dies. A bare
# `wait` reported success even when a GPU OOMed, so the sweep marched on and
# left silent holes in the results tree.
run_group() {
    local flag="$1"; shift
    local press="$1"; shift
    local -a values=("$@")
    local -a pids=()
    local device=0
    for value in "${values[@]}"; do
        "${PYTHON}" evaluate.py \
            --dataset "${DATASET}" \
            --data_dir "${DATA_DIR}" \
            --model "${MODEL}" \
            --press_name "${press}" \
            "${flag}" "${value}" \
            --output_dir "${OUTPUT_DIR}" \
            --device "cuda:${device}" \
            ${QUERY_AWARE:+--query_aware} &
        pids+=("$!")
        device=$((device + 1))
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    if [ "${failed}" -ne 0 ]; then
        echo "FAILED: press=${press} ${flag} ${values[*]}" >&2
        return 1
    fi
}

RATIOS=(0.25 0.50 0.75 0.875)

"${PYTHON}" evaluate.py --dataset "${DATASET}" --data_dir "${DATA_DIR}" --model "${MODEL}" \
    --press_name no_press --compression_ratio 0.00 --output_dir "${OUTPUT_DIR}" --device "cuda:0"

# Loop 1: presses not requiring to include the questions in the compression
for press in random knorm snapkv expected_attention streaming_llm tova \
             observed_attention qfilter pyramidkv lagkv keydiff adakv_compactor cur kvzip; do
    run_group --compression_ratio "${press}" "${RATIOS[@]}"
done

for press in kvzap_linear kvzap_mlp; do
    run_group --threshold "${press}" "${KVZAP_THRESHOLDS[@]}"
done

# DuoAttentionPress uses --head_compression_ratio (its compression_ratio is read-only and derived at runtime)
for press in duo_attention duo_attention_on_the_fly; do
    run_group --head_compression_ratio "${press}" "${RATIOS[@]}"
done

# Loop 2: presses requiring to compress questions
QUERY_AWARE=1
for press in snapkv adakv_snapkv finch chunkkv; do
    run_group --compression_ratio "${press}" "${RATIOS[@]}"
done
