#!/bin/bash
# Full native HF LOFT-32K baseline: no KVPress hooks or compression.
#SBATCH --job-name=direct-loft32-q35-gptq
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/home/rethinkingai-self/25m0820/logs/%x-%j.out
#SBATCH --error=/home/rethinkingai-self/25m0820/logs/%x-%j.err

set -euo pipefail
# Overridable for other hosts/accounts. The #SBATCH --output lines above cannot
# expand variables, so they repeat this default path literally -- change both,
# and note the log directory must already exist before the job is submitted.
BASE_DIR="${BASE_DIR:-/home/rethinkingai-self/25m0820}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/kvpress}"
# kvpress-tf515, not kvpress: the Qwen3.5 dynamic-GPTQ path these jobs load
# needs transformers==5.15.0, the same env run-eval.sh documents as required.
PYTHON="${PYTHON:-${BASE_DIR}/miniconda3/envs/kvpress-tf515/bin/python}"
MODEL_DIR="${REPO_DIR}/Qwen3.5-27B-GPTQ-Int4"
test -f "${MODEL_DIR}/config.json"
mkdir -p "${BASE_DIR}/logs"
cd "${REPO_DIR}"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" direct_baseline/run_baseline.py \
  --config direct_baseline/configs/qwen35_27b_gptq_int4_loft32k.yaml
