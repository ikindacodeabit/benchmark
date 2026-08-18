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
# Second venv for the kvpress-backed arm 4 and the kvzip vanilla baseline. The
# RLM venv above pins transformers==4.51.3 for the vLLM server; kvpress needs
# >=4.56 (new Cache API), so the two cannot share an environment. The benchmark
# DRIVER imports neither vllm nor transformers by itself, so arm 4 runs the
# driver from THIS venv (hosting the compressed sub model in-process) while the
# root still talks HTTP to the vLLM server from the other venv.
KVPRESS_VENV="${KVPRESS_VENV:-.venv-kvpress}"

# --- arm-4 colocation mode (KVPRESS_ARMS=1) -----------------------------------
# Arm 4 puts TWO models on one card: the vLLM root server plus the in-process HF
# sub model. The root never sees the document (it reads REPL observations, ~6k
# chars/turn), so its window shrinks from 139264 to KV_ROOT_MAX_LEN and its
# memory ceiling drops to leave SUB_RESERVE_MIB for the sub model. Budget on an
# empty 48 GB (49140 MiB) card:
#   vLLM at ROOT_BUDGET_MIB      -> ~21 GB (8 weights + overhead + KV pool;
#                                   one 65536-token sequence needs ~9.2 GB of KV
#                                   and vLLM refuses to start if it cannot fit)
#   HF sub reservation ~16 GB    -> 8.1 weights + 4.9 GB KV for a 34k-token call
#                                   (KVzip MASKS keys, it does not free them, so
#                                   plan for the full uncompressed KV) + ~2.5 GB
#                                   scoring transient
#   total ~37 GB, ~11 GB margin for co-tenants.
if [ -n "${KVPRESS_ARMS:-}" ]; then
    MIN_FREE_MIB="${MIN_FREE_MIB:-38000}"
else
    # 8 GB of weights plus enough KV for at least one 128k sequence.
    MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
fi
# Leave headroom so a co-tenant's job growing slightly does not OOM the server.
HEADROOM_MIB="${HEADROOM_MIB:-2000}"
SUB_RESERVE_MIB="${SUB_RESERVE_MIB:-16000}"
KV_ROOT_MAX_LEN="${KV_ROOT_MAX_LEN:-65536}"
# What the colocated root may take for itself: ~8 GB of weights, ~9.2 GB of KV
# for one KV_ROOT_MAX_LEN sequence, and ~3 GB of activations and non-torch
# overhead. Capping it stops a near-empty card from handing the root a KV pool
# far larger than its window can ever use.
ROOT_BUDGET_MIB="${ROOT_BUDGET_MIB:-21000}"

# Some infolab hosts mix A6000 (sm_86) and RTX 6000 Ada (sm_89). CUDA orders
# devices FASTEST_FIRST by default while nvidia-smi reports PCI order, so without
# this the index measured as idle is not the index CUDA selects.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Validate the subcommand first, so a typo reports itself rather than surfacing as
# a scratch or missing-activate error.
case "${1:-}" in
auto | serve | run | setup | kvzip-baseline) ;;
*)
    echo "usage: $0 {setup|auto|serve|run|kvzip-baseline}" >&2
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
    # Pin transformers LAST so it wins over both installs above. pyproject asks for
    # >=4.56 for the kvpress side, but transformers 5.x removed
    # `all_special_tokens_extended`, which vLLM 0.8.5 still calls -- the server dies
    # at tokenizer init with "Qwen2Tokenizer has no attribute
    # all_special_tokens_extended". pip will warn about the kvpress conflict; that is
    # expected and harmless here, because the RLM arm never imports the presses.
    # Consequence: this venv is for the RLM path only. Keep a separate one for kvpress.
    pip install "transformers==${TRANSFORMERS_VERSION:-4.51.3}"

    python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, devices {torch.cuda.device_count()}")
print("capabilities:", {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
import transformers
print(f"transformers {transformers.__version__}")
PY

    # Pre-fetch the weights HERE, with a visible progress bar, rather than letting
    # the first `vllm serve` download ~8 GB inside the readiness window. That window
    # is 15 minutes by default, which a cold download can easily exceed -- and the
    # failure reads as "server never became ready", which points at the wrong thing.
    echo "pre-fetching $MODEL into $HF_HOME ..."
    hf download "$MODEL" || huggingface-cli download "$MODEL"

    # Reproduce the exact call vLLM makes at tokenizer init. Without this the first
    # incompatibility only shows up ~5 minutes into a serve, as a readiness timeout
    # that points at the GPU rather than at the dependency set.
    MODEL="$MODEL" python - <<'PY'
import os
import sys

from transformers import AutoTokenizer

model = os.environ["MODEL"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
if not hasattr(tok, "all_special_tokens_extended"):
    sys.exit(
        f"INCOMPATIBLE: {type(tok).__name__} has no `all_special_tokens_extended`, "
        "which vLLM 0.8.5 calls at startup. Pin transformers==4.51.3 "
        "(TRANSFORMERS_VERSION=... to override)."
    )
print(f"tokenizer OK: {type(tok).__name__}, vocab {len(tok)}")
PY

    # --- kvpress venv (arm 4 + kvzip baseline) --------------------------------
    # Installed via explicit $KVPRESS_VENV/bin paths, not activate, so the RLM
    # venv sourced above stays the active one for the rest of setup.
    echo "--- setting up $KVPRESS_VENV (kvpress arm) ---"
    if [ ! -f "$KVPRESS_VENV/bin/activate" ]; then
        if command -v uv >/dev/null 2>&1; then
            uv venv "$KVPRESS_VENV"
        else
            python3 -m venv "$KVPRESS_VENV" || {
                python3 -m venv --without-pip "$KVPRESS_VENV"
                curl -sS https://bootstrap.pypa.io/get-pip.py | "$KVPRESS_VENV/bin/python"
            }
        fi
    fi
    # Torch FIRST and pinned: 2.6.0 ships cu124 wheels, which the infolab
    # CUDA-12.5 driver accepts (an unpinned resolve can grab a cu13x build that
    # dies at init with "driver too old"). It satisfies kvpress's >=2.3.1,<3, so
    # the editable install below keeps it.
    "$KVPRESS_VENV/bin/pip" install "torch==${KVPRESS_TORCH_VERSION:-2.6.0}"
    "$KVPRESS_VENV/bin/pip" install -e ".[eval]"
    if [ -n "${TRANSFORMERS_KVPRESS_VERSION:-}" ]; then
        # Escape hatch if a fresh transformers release breaks kvpress.
        "$KVPRESS_VENV/bin/pip" install "transformers==$TRANSFORMERS_KVPRESS_VERSION"
    fi

    # Probe the exact seams arm 4 depends on, HERE, where the error message can
    # say what to fix -- not five minutes into a GPU run.
    MODEL="$MODEL" "$KVPRESS_VENV/bin/python" - <<'PY'
import os

import torch
import transformers

print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"transformers {transformers.__version__}")

from transformers import DynamicCache

assert hasattr(DynamicCache(), "layers"), (
    "transformers too old for kvpress (no Cache.layers); "
    "need >=4.56 in this venv (TRANSFORMERS_KVPRESS_VERSION=... to override)"
)

from kvpress import KVzipPress

press = KVzipPress()
press.compression_ratio = 0.5
print(f"kvpress OK: {type(press).__name__}(compression_ratio={press.compression_ratio})")

from evaluation.rlm.kvpress_backend import SUB_PRESS_CHOICES  # import-graph check

print(f"kvpress_backend OK: presses {SUB_PRESS_CHOICES}")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ["MODEL"], trust_remote_code=True)
rendered = tok.apply_chat_template(
    [{"role": "user", "content": "probe"}],
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)
assert "probe" in rendered
print("chat template OK (tolerates enable_thinking)")
PY

    echo "setup done. Next: DATASETS=\"nq\" LENGTH=32k LIMIT=3 SERVERS=1 $0 auto"
    echo "arm 4:      KVPRESS_ARMS=1 KV_RATIOS=0.5 DATASETS=nq LENGTH=32k LIMIT=1 SERVERS=1 $0 auto"
    echo "cell 5:     DATASETS=nq KV_RATIOS=0.5 $0 kvzip-baseline"
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

# --gpu-memory-utilization is a ceiling on TOTAL DEVICE memory, and vLLM charges
# EVERY tenant's bytes against it -- ours and other users' alike. So the fraction
# is (what the co-tenants already hold) + (what we want for ourselves), over
# total. Deriving it from `free` instead subtracts the co-tenants a SECOND time,
# because `free` is already net of them, and the KV pool is what absorbs the
# shortfall. That is not theoretical: on bee 2026-08-18, util 0.44 on a card with
# ~10 GB of co-tenants left the root 1.40 GiB of KV against the 9.00 GiB one
# 65536-token sequence needs, and the engine core refused to start.
util_for() {
    local idx="$1" free total
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx")
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$idx")
    awk -v f="$free" -v t="$total" -v h="$HEADROOM_MIB" \
        'BEGIN{u=(t-f+f-h)/t; if(u>0.90)u=0.90; if(u<0.50)u=0.50; printf "%.2f", u}'
}

# Colocation variant: the root shares the card with the ~16 GB in-process sub
# model, so its share is what is free less that reservation and the headroom,
# capped at ROOT_BUDGET_MIB. Same total-device correction as above: add back what
# the co-tenants hold. The ceiling this produces leaves SUB_RESERVE_MIB +
# HEADROOM_MIB of the card unclaimed by construction, whatever the co-tenants do.
# When SUB_GPUS is set the sub model has its own card and the plain shape is used.
kv_util_for() {
    local idx="$1" free total
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx")
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$idx")
    awk -v f="$free" -v t="$total" -v h="$HEADROOM_MIB" -v s="$SUB_RESERVE_MIB" \
        -v b="$ROOT_BUDGET_MIB" \
        'BEGIN{own=f-h-s; if(own>b)own=b; u=(t-f+own)/t;
               if(u>0.90)u=0.90; if(u<0.30)u=0.30; printf "%.2f", u}'
}

serve_on() {
    local idx="$1" port="$2" util maxlen
    if [ -n "${KVPRESS_ARMS:-}" ] && [ -z "${SUB_GPUS:-}" ]; then
        # Arm-4 colocation: root shares the card with the in-process sub model.
        # The root only ever sees REPL transcripts, never the document, so a
        # smaller window is safe; an overlong transcript 400s, is recorded as an
        # error, and is retried on resume (raise KV_ROOT_MAX_LEN if it recurs).
        util=$(kv_util_for "$idx")
        maxlen="$KV_ROOT_MAX_LEN"
    else
        util=$(util_for "$idx")
        # --max-model-len 139264 = 131072 of LOFT context + headroom. Qwen3-4B-2507
        # is natively 262144, so no YaRN. No --reasoning-parser: 2507 is non-thinking.
        maxlen=139264
    fi
    echo "GPU $idx -> port $port, --max-model-len $maxlen, --gpu-memory-utilization $util"
    CUDA_VISIBLE_DEVICES="$idx" vllm serve "$MODEL" \
        --served-model-name "$MODEL" \
        --port "$port" \
        --max-model-len "$maxlen" \
        --gpu-memory-utilization "$util" \
        --enforce-eager -O0
}

# Readiness must verify WHICH model answers, not just that something does. On a
# shared host another user can already hold the port: our vllm then dies at bind
# ("address already in use"), a bare 200-check happily accepts THEIR server, and
# every request 404s with "model does not exist" -- which is exactly how this
# failed on bee. The model-name grep makes a foreign server fail readiness.
wait_ready() {
    local port="$1"
    for _ in $(seq 1 "${READY_TRIES:-90}"); do
        if curl -sf --max-time 5 "http://localhost:$port/v1/models" | grep -qF "\"$MODEL\""; then
            return 0
        fi
        sleep 10
    done
    return 1
}

# Fail FAST (not after a 15-minute readiness timeout) when the port is taken.
port_must_be_free() {
    local port="$1"
    if curl -sf -o /dev/null --max-time 2 "http://localhost:$port/v1/models" ||
        curl -s -o /dev/null --max-time 2 "http://localhost:$port/"; then
        echo "ERROR: port $port is already in use (another user's server?)." >&2
        echo "  Pick a free base port: PORT=<port> $0 ..." >&2
        return 1
    fi
    return 0
}

case "${1:-}" in
serve)
    GPU="${GPU:?set GPU=<index>; run nvidia-smi first, the cards are shared}"
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    if [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
        echo "ERROR: GPU $GPU has only ${FREE} MiB free, need ${MIN_FREE_MIB}" >&2
        exit 1
    fi
    port_must_be_free "$PORT" || exit 1
    serve_on "$GPU" "$PORT"
    ;;

run)
    for ds in $DATASETS; do
        DATASET="$ds" LENGTH="$LENGTH" PORT="$PORT" LIMIT="$LIMIT" \
            RESULTS="$RESULTS" MODEL="$MODEL" LOGS="$LOGS" \
            KVPRESS_PYTHON="$KVPRESS_VENV/bin/python" \
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

    for i in "${!GPUS[@]}"; do
        port_must_be_free $((PORT + i)) || exit 1
    done

    PIDS=()
    # Two subtleties, both learned from a stuck Ctrl-C on bee: (1) after an INT
    # trap runs, bash RESUMES the interrupted loop unless the handler exits, so
    # ^C looked acknowledged but the script kept polling forever; (2) each PID
    # here is the backgrounded subshell, and killing it orphans the vllm child
    # inside -- the children must be killed too (pkill -P).
    cleanup() {
        trap - EXIT INT TERM
        echo "shutting down servers: ${PIDS[*]-}"
        for p in "${PIDS[@]-}"; do
            pkill -TERM -P "$p" 2>/dev/null || true
            kill "$p" 2>/dev/null || true
        done
        exit "${1:-0}"
    }
    trap 'cleanup 130' INT TERM
    trap 'cleanup $?' EXIT

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

    # In arm-4 mode the sub model colocates with its lane's server by default;
    # SUB_GPUS="i j ..." instead deals dedicated sub cards to the lanes (and the
    # servers then keep the full 139264 window).
    read -r -a SUB_ARR <<<"${SUB_GPUS:-}"

    WPIDS=()
    for i in "${!GPUS[@]}"; do
        p=$((PORT + i))
        sub_gpu="${GPUS[$i]}"
        if [ "${#SUB_ARR[@]}" -gt 0 ]; then
            sub_gpu="${SUB_ARR[$((i % ${#SUB_ARR[@]}))]}"
        fi
        (
            for j in "${!DS_ARR[@]}"; do
                # Deal datasets round-robin so each server gets a fair share.
                if [ $((j % ${#GPUS[@]})) -eq "$i" ]; then
                    DATASET="${DS_ARR[$j]}" LENGTH="$LENGTH" PORT="$p" LIMIT="$LIMIT" \
                        RESULTS="$RESULTS" MODEL="$MODEL" LOGS="$LOGS" \
                        KVPRESS_PYTHON="$KVPRESS_VENV/bin/python" \
                        RLM_SUB_GPU="$sub_gpu" \
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

kvzip-baseline)
    # Cell 5: vanilla model + KVzip through the standard kvpress evaluate.py
    # path -- the press-only baseline the RLM arms are compared against. Pure
    # launcher: all logic already exists in evaluation/evaluate.py. LOFT shares
    # ONE corpus per subset, so each (dataset, ratio) pays a single ~119k-token
    # KVzip prefill (KVzip scores in extra passes, expect 10-20 min) amortized
    # over all questions. Resumable: evaluate.py skips completed run dirs.
    if [ ! -f "$KVPRESS_VENV/bin/python" ]; then
        echo "ERROR: no venv at $KVPRESS_VENV. Run '$0 setup' first." >&2
        exit 1
    fi
    KVPY_ABS="$(cd "$KVPRESS_VENV" && pwd)/bin/python"
    # One 119k-token prefill needs ~17 GB of KV on top of ~8 GB of weights.
    BASELINE_MIN_FREE_MIB="${BASELINE_MIN_FREE_MIB:-33000}"
    MIN_FREE_MIB="$BASELINE_MIN_FREE_MIB"
    GPU_PICK="$(pick_gpus 1)"
    if [ -z "$GPU_PICK" ]; then
        echo "ERROR: no GPU has ${MIN_FREE_MIB} MiB free. Current state:" >&2
        nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv >&2
        exit 1
    fi
    echo "kvzip baseline on GPU $GPU_PICK"
    OUT_ABS="$(pwd)/${KV_BASELINE_RESULTS:-evaluation/results/loft128k_kvpress}"
    mkdir -p "$OUT_ABS"
    FAILED=0
    for ds in $DATASETS; do
        for R in ${KV_RATIOS:-0.5 0.75}; do
            echo "=== $(date '+%F %T') :: kvzip baseline :: ${ds}_${LENGTH} ratio $R ==="
            # evaluate.py's imports are flat (from benchmarks... import), so it
            # must run with cwd=evaluation/.
            (
                cd evaluation &&
                    CUDA_VISIBLE_DEVICES="$GPU_PICK" "$KVPY_ABS" evaluate.py \
                        --dataset loft \
                        --data_dir "${ds}_${LENGTH}" \
                        --model "$MODEL" \
                        --press_name "${KV_PRESS:-kvzip}" \
                        --compression_ratio "$R" \
                        --device cuda:0 \
                        --fraction "${FRACTION:-1.0}" \
                        --output_dir "$OUT_ABS"
            ) || {
                echo "WARN: kvzip baseline ${ds}_${LENGTH} ratio $R failed" >&2
                FAILED=1
            }
        done
    done
    exit "$FAILED"
    ;;

*)
    echo "usage: $0 {setup|auto|serve|run|kvzip-baseline}" >&2
    exit 2
    ;;
esac
