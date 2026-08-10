#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# RLM evaluation with the comparability fixes and runaway guards enabled.
# Phase 1: all inference goes to the NIM cloud API, so this needs no GPU.
#
#SBATCH --job-name=rlm-comparable
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/home/rethinkingai-self/gautam_mahale/logs/%x-%j.out
#SBATCH --error=/home/rethinkingai-self/gautam_mahale/logs/%x-%j.err

set -euo pipefail

BASE_DIR=/home/rethinkingai-self/gautam_mahale
REPO_DIR="${BASE_DIR}/kvpress"

mkdir -p "${BASE_DIR}/logs"
cd "${REPO_DIR}"

source .venv/bin/activate

export HF_HOME="${HF_HOME:-${BASE_DIR}/hf_cache}"
export RLM_DATA_DIR="${RLM_DATA_DIR:-${BASE_DIR}/rlm_data}"
export TOKENIZERS_PARALLELISM=false

: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY before submitting}"

# --- fail fast if this compute node has no outbound internet ---
curl -s -o /dev/null -w "NIM API reachability: HTTP %{http_code}\n" --max-time 20 \
  https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}" \
  || { echo "ERROR: no outbound internet from $(hostname) — ask hpc@iitb.ac.in about a proxy"; exit 1; }

# Set to (max-model-len - max-tokens - margin) for the root model. Without this,
# densely-tokenising subsets (RULER cwe, niah_multikey_3) overflow the served
# window and score 0.0 from a harness error rather than from the model.
VANILLA_MAX_PROMPT_TOKENS=120000

# --- sequential grid: one process, full --rpm to itself ---
for TASK in niah multikey longbench_v2; do
  for MODE in vanilla rlm; do
    echo "=== $(date '+%F %T') :: ${TASK} / ${MODE} ==="
    python -m evaluation.rlm.run_benchmark \
      --dataset "${TASK}" --mode "${MODE}" \
      --limit 50 --rpm 30 \
      --vanilla-max-prompt-tokens "${VANILLA_MAX_PROMPT_TOKENS}" \
      --exec-timeout 60 \
      --run-timeout 900 \
      --max-sub-calls 40 \
      --out evaluation/results/rlm \
      || echo "WARN: ${TASK}/${MODE} failed, continuing with next cell"
  done
done

echo "=== $(date '+%F %T') :: all cells done ==="
