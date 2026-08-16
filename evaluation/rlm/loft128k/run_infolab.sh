#!/bin/bash
# LOFT-128k on the CSE infolab hosts (bee/fox.cse.iitb.ac.in) -- the primary path.
#
# These hosts have NO scheduler, and their GPUs are shared and unreserved: nothing
# restarts a dead run, nothing stops another user claiming a card mid-setup, and a
# card that measured idle a minute ago may not be. Everything here is built around
# that: pick cards by live utilisation, size the KV pool from what is ACTUALLY
# free, and hold a per-dataset lock so a relaunch cannot duplicate work.
#
# Always run under tmux -- an ssh drop otherwise kills the servers and the workers.
#
#   tmux new -s rlm
#   bash evaluation/rlm/loft128k/run_infolab.sh auto          # full grid
#
# Or drive the two halves by hand, in two tmux windows:
#   GPU=1 bash evaluation/rlm/loft128k/run_infolab.sh serve
#   DATASETS="nq" LIMIT=3 LENGTH=32k bash evaluation/rlm/loft128k/run_infolab.sh run
#
# SIZING on a 48 GB card. Qwen3-4B-2507 is 36 layers x 8 KV heads x 128 head_dim
# = 144 KiB per token, so one 128k sequence needs ~18 GB of KV on top of ~8 GB of
# bf16 weights. On an EMPTY card (util cap 0.85 -> ~41 GB) that is roughly 1.8
# concurrent sequences. On a card already holding another user's 13 GB job the cap
# lands near 0.59 -> ~29 GB, which fits the weights and barely ONE 128k sequence.
# It still runs; it is just slow. Prefer the emptiest cards.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-8000}"
LENGTH="${LENGTH:-128k}"
LIMIT="${LIMIT:-110}"
DATASETS="${DATASETS:-nq hotpotqa musique qampari quest}"
RESULTS="${RESULTS:-evaluation/results/loft128k}"
LOGS="${LOGS:-$RESULTS/logs}"
VENV="${VENV:-.venv}"
# 8 GB of weights plus enough KV for at least one 128k sequence.
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
# Leave headroom so a co-tenant's job growing slightly does not OOM the server.
HEADROOM_MIB="${HEADROOM_MIB:-2000}"

# Some infolab hosts mix A6000 (sm_86) and RTX 6000 Ada (sm_89). CUDA orders
# devices FASTEST_FIRST by default while nvidia-smi reports PCI order, so without
# this the index measured as idle is not the index CUDA selects.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Validate the subcommand first, so a typo reports itself rather than surfacing as
# a scratch or missing-activate error.
case "${1:-}" in
auto | serve | run | setup) ;;
*)
    echo "usage: $0 {setup|auto|serve|run}" >&2
    exit 2
    ;;
esac

# --- keep caches off $HOME ----------------------------------------------------
# $HOME is a small NFS quota on these hosts and `quota -u` prints nothing, so the
# cap is invisible and `df` is misleading. Model weights (~8 GB) plus the LOFT
# parquet will blow through it, and a large write to a stalled filer mount is
# uninterruptible (hard,timeo=600 -> D-state, Ctrl-C does nothing) and has taken
# the home directory down host-wide before. Everything cacheable is redirected.
RLM_SCRATCH="${RLM_SCRATCH:-/mnt/nas/$USER}"
if [ ! -d "$RLM_SCRATCH" ]; then
    echo "WARNING: RLM_SCRATCH=$RLM_SCRATCH does not exist." >&2
    echo "  Caches would fall back to \$HOME, which is quota-capped on these hosts." >&2
    echo "  Set RLM_SCRATCH=<a big writable path> before running, or create that dir." >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$RLM_SCRATCH/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RLM_SCRATCH/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

mkdir -p "$RESULTS" "$LOGS"

# --- setup --------------------------------------------------------------------
# Runs AFTER the cache redirection above, which is the whole point: `uv sync` and
# `pip install` write several GB of wheels and unpacked packages, and their caches
# default to $HOME. Doing the install through this subcommand is what keeps that
# off the quota-capped home directory.
if [ "$1" = "setup" ]; then
    echo "scratch:  $RLM_SCRATCH"
    echo "HF_HOME:  $HF_HOME"
    echo "caches:   $XDG_CACHE_HOME"
    df -BG "$RLM_SCRATCH" | tail -1

    if [ ! -f "$VENV/bin/activate" ]; then
        if command -v uv >/dev/null 2>&1; then
            uv venv "$VENV"
        else
            # Not every infolab host has uv, and some Debian pythons ship without
            # ensurepip.
            python3 -m venv "$VENV" || {
                python3 -m venv --without-pip "$VENV"
                # shellcheck disable=SC1091
                source "$VENV/bin/activate"
                curl -sS https://bootstrap.pypa.io/get-pip.py | python3
            }
        fi
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"

    pip install -e ".[eval]"
    # Pin vLLM: an unpinned install resolves to a build whose torch targets a newer
    # CUDA than some infolab drivers accept, and 0.8.5.post1 is the version the
    # earlier Qwen3-8B numbers were produced with, so results stay comparable.
    pip install "vllm==${VLLM_VERSION:-0.8.5.post1}"

    python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, devices {torch.cuda.device_count()}")
print("capabilities:", {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
PY

    # Pre-fetch the weights HERE, with a visible progress bar, rather than letting
    # the first `vllm serve` download ~8 GB inside the readiness window. That window
    # is 15 minutes by default, which a cold download can easily exceed -- and the
    # failure reads as "server never became ready", which points at the wrong thing.
    echo "pre-fetching $MODEL into $HF_HOME ..."
    hf download "$MODEL" || huggingface-cli download "$MODEL"

    echo "setup done. Next: DATASETS=\"nq\" LENGTH=32k LIMIT=3 SERVERS=1 $0 auto"
    exit 0
fi

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: no venv at $VENV. Run '$0 setup' first, or set VENV=<path>." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- GPU selection ------------------------------------------------------------
# Rank by UTILISATION, not free memory. Ranking by free memory alone picks a card
# that is idle-looking but pegged at 99% SM by a small-footprint job.
pick_gpus() {
    local want="$1"
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F', *' -v m="$MIN_FREE_MIB" '$2 >= m { print $3, -$2, $1 }' |
        sort -n -k1,1 -k2,2 | head -n "$want" | awk '{print $3}'
}

# --gpu-memory-utilization is a ceiling on TOTAL DEVICE memory, not on our own
# allocation, so it must be derived from total-minus-other-tenants, not from free.
util_for() {
    local idx="$1" free total
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx")
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$idx")
    awk -v f="$free" -v t="$total" -v h="$HEADROOM_MIB" \
        'BEGIN{u=(f-h)/t; if(u>0.85)u=0.85; if(u<0.50)u=0.50; printf "%.2f", u}'
}

serve_on() {
    local idx="$1" port="$2" util
    util=$(util_for "$idx")
    echo "GPU $idx -> port $port, --gpu-memory-utilization $util"
    # --max-model-len 139264 = 131072 of LOFT context + headroom. Qwen3-4B-2507 is
    # natively 262144, so no YaRN. No --reasoning-parser: 2507 is non-thinking.
    CUDA_VISIBLE_DEVICES="$idx" vllm serve "$MODEL" \
        --served-model-name "$MODEL" \
        --port "$port" \
        --max-model-len 139264 \
        --gpu-memory-utilization "$util" \
        --enforce-eager -O0
}

wait_ready() {
    local port="$1"
    for _ in $(seq 1 "${READY_TRIES:-90}"); do
        curl -sf -o /dev/null --max-time 5 "http://localhost:$port/v1/models" && return 0
        sleep 10
    done
    return 1
}

case "${1:-}" in
serve)
    GPU="${GPU:?set GPU=<index>; run nvidia-smi first, the cards are shared}"
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    if [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
        echo "ERROR: GPU $GPU has only ${FREE} MiB free, need ${MIN_FREE_MIB}" >&2
        exit 1
    fi
    serve_on "$GPU" "$PORT"
    ;;

run)
    for ds in $DATASETS; do
        DATASET="$ds" LENGTH="$LENGTH" PORT="$PORT" LIMIT="$LIMIT" \
            RESULTS="$RESULTS" MODEL="$MODEL" LOGS="$LOGS" \
            bash evaluation/rlm/loft128k/run_cells.sh
    done
    ;;

auto)
    # One server per usable card, datasets dealt round-robin across them. A 4B
    # model needs no tensor parallelism -- N independent servers is the right way
    # to use N cards here.
    read -r -a DS_ARR <<<"$DATASETS"
    WANT="${SERVERS:-${#DS_ARR[@]}}"
    read -r -a GPUS <<<"$(pick_gpus "$WANT" | tr '\n' ' ')"

    if [ "${#GPUS[@]}" -eq 0 ]; then
        echo "ERROR: no GPU has ${MIN_FREE_MIB} MiB free. Current state:" >&2
        nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv >&2
        exit 1
    fi
    echo "using GPUs: ${GPUS[*]} for ${#DS_ARR[@]} datasets"

    PIDS=()
    cleanup() {
        echo "shutting down servers: ${PIDS[*]-}"
        for p in "${PIDS[@]-}"; do kill "$p" 2>/dev/null || true; done
    }
    trap cleanup EXIT INT TERM

    for i in "${!GPUS[@]}"; do
        p=$((PORT + i))
        serve_on "${GPUS[$i]}" "$p" >"$LOGS/vllm.gpu${GPUS[$i]}.log" 2>&1 &
        PIDS+=($!)
    done

    for i in "${!GPUS[@]}"; do
        p=$((PORT + i))
        if ! wait_ready "$p"; then
            echo "ERROR: server on port $p never became ready; see $LOGS/vllm.gpu${GPUS[$i]}.log" >&2
            exit 1
        fi
        echo "server ready on port $p"
    done

    WPIDS=()
    for i in "${!GPUS[@]}"; do
        p=$((PORT + i))
        (
            for j in "${!DS_ARR[@]}"; do
                # Deal datasets round-robin so each server gets a fair share.
                if [ $((j % ${#GPUS[@]})) -eq "$i" ]; then
                    DATASET="${DS_ARR[$j]}" LENGTH="$LENGTH" PORT="$p" LIMIT="$LIMIT" \
                        RESULTS="$RESULTS" MODEL="$MODEL" LOGS="$LOGS" \
                        bash evaluation/rlm/loft128k/run_cells.sh || true
                fi
            done
        ) &
        WPIDS+=($!)
    done

    FAILED=0
    for w in "${WPIDS[@]}"; do wait "$w" || FAILED=1; done
    echo "=== $(date '+%F %T') :: all datasets done (failed=$FAILED) ==="
    exit "$FAILED"
    ;;

*)
    echo "usage: $0 {auto|serve|run}" >&2
    exit 2
    ;;
esac
