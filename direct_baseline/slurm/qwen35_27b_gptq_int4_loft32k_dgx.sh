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
BASE_DIR=/home/rethinkingai-self/25m0820
REPO_DIR="${BASE_DIR}/kvpress"
MODEL_DIR="${REPO_DIR}/Qwen3.5-27B-GPTQ-Int4"
test -f "${MODEL_DIR}/config.json"
mkdir -p "${BASE_DIR}/logs"
cd "${REPO_DIR}"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${BASE_DIR}/miniconda3/envs/kvpress/bin/python" direct_baseline/run_baseline.py \
  --config direct_baseline/configs/qwen35_27b_gptq_int4_loft32k.yaml
