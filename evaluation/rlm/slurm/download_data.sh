#!/bin/bash
# Run on the LOGIN node (needs internet). Populate the shared Hugging Face cache.
# Usage (from repository root): bash evaluation/rlm/slurm/download_data.sh
#
# Compute nodes are air-gapped and run with HF_HUB_OFFLINE=1, so anything not
# cached here fails at load time inside the job. LOFT_LENGTHS selects which LOFT
# splits to pull; the default is the 128k grid. `1m` is large and opt-in.
#   LOFT_LENGTHS="32k 128k" bash evaluation/rlm/slurm/download_data.sh
#
# STAGE selects which benchmark groups to cache (default: all three), so a run
# that only needs one is not blocked by another's upstream layout changing:
#   STAGE=loft LOFT_LENGTHS="1m" bash evaluation/rlm/slurm/download_data.sh
set -euo pipefail

LOFT_LENGTHS="${LOFT_LENGTHS:-128k}"
export LOFT_LENGTHS

# Honour a pre-set HF_HOME/VENV instead of overriding them. Both used to be
# hardcoded to $HOME, which is a small quota-capped NFS mount on the infolab
# hosts -- staging LOFT there fails with "OSError: [Errno 122] Disk quota
# exceeded" partway through, and a repo-relative .venv does not exist when the
# venv lives on scratch. Defaults match run_infolab.sh's RLM_SCRATCH layout.
RLM_SCRATCH="${RLM_SCRATCH:-/mnt/nas/$USER}"
export HF_HOME="${HF_HOME:-$RLM_SCRATCH/hf_cache}"
case "$HF_HOME" in
"$HOME"/*)
    echo "ERROR: HF_HOME=$HF_HOME is under \$HOME, which is quota-capped here." >&2
    echo "  Set HF_HOME (or RLM_SCRATCH) to a big writable path before running." >&2
    exit 1
    ;;
esac
mkdir -p "$HF_HOME"

VENV="${VENV:-.venv}"
if [ ! -x "$VENV/bin/activate" ] && [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: no venv at $VENV; set VENV=<path to your venv>" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python - <<'EOF'
import os

from datasets import load_dataset
from evaluation.benchmarks.registry import DATASET_REGISTRY, LOFT_RAG_DATASETS, RULER_32K_TASKS

# Which groups to cache. Staging everything unconditionally means one unrelated
# benchmark's upstream layout change blocks the group you actually wanted --
# e.g. a LOFT-only run dying inside the RULER loop.
stage = set(os.environ.get("STAGE", "longbench ruler loft").split())

# These are the same IDs/configs consumed by KVPress's shared loader.
if "longbench" in stage:
    print("Caching shared LongBench-v2 test split ...")
    longbench = load_dataset(DATASET_REGISTRY["longbench-v2"], split="test")
    print(f"LongBench-v2: {len(longbench)} examples")

if "ruler" in stage:
    print("Caching shared RULER-32K subsets ...")
    for task in RULER_32K_TASKS:
        # Must match _load_ruler32k: each task is its own CONFIG and its own
        # SPLIT, both named after the task. Passing data_dir= instead makes
        # datasets treat the repo as a directory of raw files and synthesise a
        # single "train" split, so split="test" then fails with
        # 'Unknown split "test". Should be one of [train]'.
        dataset = load_dataset(DATASET_REGISTRY["ruler32k"], task, split=task)
        print(f"RULER-32K/{task}: {len(dataset)} examples")

# --- LOFT ---
# LOFT is mirrored per (dataset, length) as its own repo rather than as configs of
# one repo, so each combination is a separate load_dataset call. `_load_loft`
# concatenates dev+test, so both splits must be cached.
if "loft" in stage:
    lengths = os.environ.get("LOFT_LENGTHS", "128k").split()
    print(f"Caching LOFT RAG subsets for lengths: {', '.join(lengths)} ...")
    for length in lengths:
        for name in LOFT_RAG_DATASETS:
            repo = f"f20180301/loft-rag-{name}-{length}"
            splits = load_dataset(repo)
            counts = ", ".join(f"{s}={len(d)}" for s, d in splits.items())
            print(f"LOFT/{name}_{length}: {counts}")

# --- OOLONG ---
# The OOLONG benchmark splits live on the Hugging Face Hub; the repo/config
# names have changed since release — check https://huggingface.co/datasets?search=oolong
# and the official RLM repo (github.com/alexzhang13/rlm) for the exact loader,
# then write oolong.jsonl under RLM_DATA_DIR with fields:
#   {"id", "context", "question", "answers": [...]}.
EOF

echo "Done. Shared datasets cached under $HF_HOME"
