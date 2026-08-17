#!/bin/bash
# Download and validate the 32K Synthetic-KV dataset.
set -euo pipefail

PRAJNA_BASE=/home/rethinkingai-self/25m0820
PRAJNA_REPO="${PRAJNA_BASE}/kvpress"
PRAJNA_PYTHON="${PRAJNA_BASE}/miniconda3/envs/kvpress/bin/python"
export HF_HOME="${PRAJNA_BASE}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=false
export HF_DATASETS_OFFLINE=false

test -x "${PRAJNA_PYTHON}"
mkdir -p "${HF_DATASETS_CACHE}"
"${PRAJNA_PYTHON}" "${PRAJNA_REPO}/evaluation/benchmarks/synthetic32k/prepare_dataset.py"

export HF_HUB_OFFLINE=true
export HF_DATASETS_OFFLINE=true
"${PRAJNA_PYTHON}" "${PRAJNA_REPO}/evaluation/benchmarks/synthetic32k/prepare_dataset.py" --offline-check
