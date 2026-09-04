#!/bin/bash
# Run the 4GB x 8x LOFT-1m cells (N=217,008) on ant, as a good tenant.
#
# The cell needs ~68 GiB free before weights load (8 GiB weights + 2.0 x 29.8 GiB
# of KVzip cache and scoring transients), so it fits nowhere but ant's 96 GB
# cards. That is a big, long hold on a shared machine -- roughly 40 min/example
# here, since the Max-Q Blackwells run this workload ~2.5x slower than an A6000 --
# so the whole point of this script is to take a card ONLY when doing so costs
# nobody else anything.
#
# Three rules, in order of how much they matter:
#
#   1. IDLE, not merely free. A card can show 72 GiB free while another user is
#      actively computing in the other 24. Taking it would leave them one bad
#      allocation from an OOM that is our fault. So: no other user's compute
#      process on the card, at all.
#   2. One cell per acquisition. The three subsets are three separate waits, and
#      the card is released between them. A 4-day unbroken hold on a shared card
#      is not defensible when the same work can leave gaps someone else can use.
#   3. Headroom. Require 72 GiB rather than the ~68 we need, so we are not the
#      reason the next allocation on that card fails.
#
# Cancel with: ssh ant.cse.iitb.ac.in 'tmux kill-session -t wait217k'
set -uo pipefail
BASE=/mnt/nas/gautammahale
NEED_GIB=${NEED_GIB:-72}
ROOT=${ROOT_HOST:-fox.cse.iitb.ac.in}
ME=$(id -un)

# Index of an idle card with enough room, or empty. "Idle" means no compute
# process belonging to anyone else; our own leftovers do not disqualify a card.
find_idle_gpu() {
    local busy uuid idx pid owner
    busy=""
    while IFS=, read -r uuid pid _; do
        pid=$(echo "$pid" | tr -d ' ')
        owner=$(ps -o user= -p "$pid" 2>/dev/null)
        [ -z "$owner" ] && continue
        [ "$owner" = "$ME" ] && continue
        idx=$(nvidia-smi --query-gpu=index --format=csv,noheader --id="$(echo "$uuid" | tr -d ' ')" 2>/dev/null | tr -d ' ')
        [ -n "$idx" ] && busy="$busy $idx"
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)

    while IFS=, read -r idx free; do
        idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
        case " $busy " in *" $idx "*) continue ;; esac
        if [ "$(( free / 1024 ))" -ge "$NEED_GIB" ]; then echo "$idx"; return 0; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    return 1
}

for subset in nq hotpotqa musique; do
    run="loft__${subset}_1m__Qwen_Qwen3-4B-Instruct-2507__rlm__split-test__scratchpad__kvzip-kvzip27126tokens__autosubx8__fixed"
    if [ -f "$BASE/fixedgrid/loft1m_grid/$run/metrics.json" ]; then
        echo "=== $(date -Is) $subset already complete, skipping ==="
        continue
    fi
    echo "=== $(date -Is) waiting for an idle card with >= ${NEED_GIB} GiB for $subset ==="
    while true; do
        if gpu=$(find_idle_gpu); then
            echo "=== $(date -Is) GPU$gpu is idle and has room; running $subset ==="
            # expandable_segments cuts fragmentation, so we ask the driver for
            # less than we otherwise would and give a bit back between calls.
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            BUDGET=27126 FACTOR=8 GPU="$gpu" LIMIT=55 \
            DATASETS="$subset" ROOT_HOST="$ROOT" \
            KVZIP_PYTHON=$BASE/benchmark/.venv-kvpress-cu128/bin/python \
            RESULTS=$BASE/fixedgrid/loft1m_grid \
            nice -n 10 bash $BASE/fixedgrid/loft_lane.sh "ant8x-$subset"
            echo "=== $(date -Is) $subset done; releasing GPU$gpu ==="
            break
        fi
        echo "$(date -Is) waiting ($subset): $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '{printf "GPU%s:%.0f ", $1, $2/1024}')"
        sleep 300
    done
done
echo "=== $(date -Is) ALL 4GB x 8x CELLS DONE ==="
