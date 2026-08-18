#!/bin/bash
# Generic sbatch script for any run_matrix.py evaluation job. Pass
# partition/qos/time/array/job-name via `sbatch` CLI flags (they override the
# #SBATCH defaults below) and pass the yml/profile/worker-count/extra flags as
# script arguments. Never needs editing -- one file for every job.
#
# Usage:
#   sbatch [--job-name=NAME --partition=P --qos=Q --time=T --array=0-N] \
#       run-eval.sh <config-file> <profile> <worker-count> [extra run_matrix.py args...]
#
# <config-file> can be "" or "default" to fall back to evaluate_config.yaml
# (Qwen3.5-27B-GPTQ-Int4, synthetic-kv 64k, max_new_tokens/max_context_length
# both null, fraction 1.0). <profile>/<worker-count> are still required
# positionally.
#
# Examples:
#   # single-shot job (loft32k, one worker owns all 5 tasks)
#   sbatch --job-name=loft32k --partition=dgx --qos=dgx --time=2-00:00:00 \
#       run-eval.sh yml/evaluate_loft32k_qwen35_27b_gptq_all_budgets.yaml \
#       loft32k-qwen35-27b-gptq-all-budgets 1
#
#   # 3-worker array job (ruler64k, sharded across a Slurm array)
#   sbatch --job-name=ruler64k --partition=dgx --qos=dgx --time=2-00:00:00 --array=0-2 \
#       run-eval.sh yml/evaluate_ruler64k_qwen35_27b_gptq_all_budgets.yaml ruler64k 3
#
#   # smoke test (single task, single budget, small fraction)
#   sbatch --job-name=ruler64k-smoke --partition=dgx --qos=dgx --time=02:00:00 \
#       run-eval.sh yml/evaluate_ruler64k_qwen35_27b_gptq_all_budgets.yaml ruler64k 1 \
#       --max-tasks=1 --max-memory-budgets=1 --fraction=0.02
#
#SBATCH --job-name=kvpress-eval
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/rethinkingai-self/25m0820/logs/%x-%j.out
#SBATCH --error=/home/rethinkingai-self/25m0820/logs/%x-%j.err

set -euo pipefail

CONFIG_FILE="$1"; shift
PROFILE="$1"; shift
WORKER_COUNT="$1"; shift
EXTRA_ARGS=("$@")

DEFAULT_CONFIG_FILE="evaluate_config.yaml"
if [ -z "${CONFIG_FILE}" ] || [ "${CONFIG_FILE}" == "default" ]; then
    CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
fi

BASE_DIR=/home/rethinkingai-self/25m0820
EVAL_DIR="${BASE_DIR}/kvpress/evaluation"
# kvpress-tf515: transformers==5.15.0 (required for the Qwen3.5 dynamic-GPTQ
# path) + kernels>=0.12.2. evaluate.py also sets checkpoint_format=gptq_v2
# for GPTQ checkpoints missing it, to skip GPTQModel's v1->v2 zero-point
# "fix" that was corrupting every quantized weight.
PYTHON="${BASE_DIR}/miniconda3/envs/kvpress-tf515/bin/python"

mkdir -p "${BASE_DIR}/logs"
cd "${EVAL_DIR}"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"

"${PYTHON}" run_matrix.py \
    --config-file="${CONFIG_FILE}" \
    --profile="${PROFILE}" \
    --worker-id="${WORKER_ID}" \
    --worker-count="${WORKER_COUNT}" \
    "${EXTRA_ARGS[@]}"
