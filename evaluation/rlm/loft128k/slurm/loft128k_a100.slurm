#!/bin/bash
# LOFT-128k: vanilla vs RLM vs RLM+scratchpad, Qwen3-4B-Instruct-2507 on Prajna.
#
# UNVERIFIED / NOT THE ACTIVE PATH. Prajna access was unavailable when this was
# written, so this script has never been submitted. The infolab hosts are the
# supported route -- see ../run_infolab.sh. This is kept so the SLURM path is
# ready if Prajna access is restored; treat it as a starting point, not a
# known-good script, and re-check the partition and paths before trusting it.
#
# One array task per LOFT dataset, each with its own single-GPU vLLM server. A 4B
# model does not need tensor parallelism -- the value of a DGX node here is five
# INDEPENDENT servers running the grid data-parallel, not sharding one model.
#
#   sbatch --partition=<dgx> --qos=<dgx> evaluation/rlm/loft128k/slurm/loft128k_a100.slurm
#
# PARTITION: run `sinfo -s` first. The directives below default to a40, which is
# known to exist and fits (48 GB is enough for 4B + one 128k sequence), but the
# A100-80GB partition is the intended target: at util 0.90 it leaves ~64 GB of KV
# pool, about 3.5 concurrent 128k sequences against roughly 1.8 on a 48 GB card.
# A command-line --partition/--qos overrides these directives.
#
#SBATCH --job-name=rlm-loft128k
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rlm-loft128k.%A_%a.out
#SBATCH --error=logs/rlm-loft128k.%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

DATASETS=(nq hotpotqa musique qampari quest)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
LIMIT="${LIMIT:-110}"
RESULTS="${RESULTS:-evaluation/results/loft128k}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export RLM_DATA_DIR="${RLM_DATA_DIR:-$HOME/rlm_data}"
# Compute nodes have no outbound internet. Everything must already be in HF_HOME
# via `bash evaluation/rlm/slurm/download_data.sh` on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

mkdir -p logs "$RESULTS"

# --- port selection -----------------------------------------------------------
# Array tasks can land on the same node, so the job id alone is not enough to keep
# their servers apart; the task id is folded in. The probe then walks past anything
# already listening, including other users' jobs.
PORT="${PORT:-$((10000 + (SLURM_ARRAY_JOB_ID % 15000) + SLURM_ARRAY_TASK_ID * 100))}"
while (echo >"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
    PORT=$((PORT + 1))
done
echo "task $SLURM_ARRAY_TASK_ID ($DATASET) -> port $PORT on $(hostname)"

# --- serve --------------------------------------------------------------------
# --max-model-len 139264 = 131072 of LOFT context + headroom for the prompt
# scaffold and the answer. Qwen3-4B-Instruct-2507 is natively 262144, so unlike
# the Qwen3-8B LOFT-128k runs this needs NO YaRN rope scaling.
# No --reasoning-parser: 2507 is a non-thinking model.
vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --max-model-len 139264 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --enforce-eager -O0 \
    >"logs/vllm.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# --- run ----------------------------------------------------------------------
DATASET="$DATASET" LENGTH=128k PORT="$PORT" LIMIT="$LIMIT" \
    RESULTS="$RESULTS" MODEL="$MODEL" \
    bash evaluation/rlm/loft128k/run_cells.sh

echo "=== $(date '+%F %T') :: array task $SLURM_ARRAY_TASK_ID ($DATASET) complete ==="
