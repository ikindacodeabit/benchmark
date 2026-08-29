#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sweep presses x compression ratios, one press per GPU, locally (no scheduler).
# Run from evaluation/. Override any setting from the environment, e.g.
#   DATASET=loft DATA_DIR=nq_128k ./evaluate.sh
set -euo pipefail

DATASET="${DATASET:-ruler}"
DATA_DIR="${DATA_DIR:-4096}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PYTHON="${PYTHON:-python}"
read -r -a COMPRESSION_RATIOS <<< "${COMPRESSION_RATIOS:-0.1 0.25 0.5}"
read -r -a PRESS_NAMES <<< "${PRESS_NAMES:-expected_attention knorm streaming_llm snapkv}"

# Count only the GPUs this process may use. Under SLURM (or any
# CUDA_VISIBLE_DEVICES mask) nvidia-smi still lists every GPU on the node, so
# the unmasked count dispatched work to cuda:1..3 the job did not own.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _visible <<< "${CUDA_VISIBLE_DEVICES}"
    num_gpus="${#_visible[@]}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    num_gpus="$(nvidia-smi --list-gpus | wc -l)"
else
    echo "Error: no CUDA_VISIBLE_DEVICES set and nvidia-smi is unavailable" >&2
    exit 1
fi

if [ "${#PRESS_NAMES[@]}" -gt "${num_gpus}" ]; then
    echo "Error: The number of press names (${#PRESS_NAMES[@]}) exceeds the number of available GPUs (${num_gpus})" >&2
    exit 1
fi

# Each press gets its own GPU; ratios run sequentially within it. Failures are
# collected rather than ignored -- `wait` alone always reported success, so an
# OOM on one GPU still printed "All evaluations completed".
pids=()
for i in "${!PRESS_NAMES[@]}"; do
    press="${PRESS_NAMES[$i]}"
    (
        for compression_ratio in "${COMPRESSION_RATIOS[@]}"; do
            echo "Running press_name: ${press} with compression_ratio: ${compression_ratio} on GPU cuda:${i}"
            "${PYTHON}" evaluate.py \
                --dataset "${DATASET}" \
                --data_dir "${DATA_DIR}" \
                --model "${MODEL}" \
                --press_name "${press}" \
                --compression_ratio "${compression_ratio}" \
                --device "cuda:${i}"
        done
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done

if [ "${failed}" -ne 0 ]; then
    echo "One or more evaluations FAILED; see the output above." >&2
    exit 1
fi
echo "All evaluations completed."
