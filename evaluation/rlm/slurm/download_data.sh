#!/bin/bash
# Run on the LOGIN node (needs internet). Populate the shared Hugging Face cache.
# Usage (from repository root): bash evaluation/rlm/slurm/download_data.sh
#
# Compute nodes are air-gapped and run with HF_HUB_OFFLINE=1, so anything not
# cached here fails at load time inside the job. LOFT_LENGTHS selects which LOFT
# splits to pull; the default is the 128k grid. `1m` is large and opt-in.
#   LOFT_LENGTHS="32k 128k" bash evaluation/rlm/slurm/download_data.sh
set -euo pipefail

LOFT_LENGTHS="${LOFT_LENGTHS:-128k}"
export LOFT_LENGTHS

export HF_HOME="$HOME/hf_cache"

source .venv/bin/activate

python - <<'EOF'
import os

from datasets import load_dataset
from evaluation.benchmarks.registry import DATASET_REGISTRY, LOFT_RAG_DATASETS, RULER_32K_TASKS

# These are the same IDs/configs consumed by KVPress's shared loader.
print("Caching shared LongBench-v2 test split ...")
longbench = load_dataset(DATASET_REGISTRY["longbench-v2"], split="test")
print(f"LongBench-v2: {len(longbench)} examples")

print("Caching shared RULER-32K subsets ...")
for task in RULER_32K_TASKS:
    dataset = load_dataset(
        DATASET_REGISTRY["ruler32k"],
        data_dir=task,
        split="test",
    )
    print(f"RULER-32K/{task}: {len(dataset)} examples")

# --- LOFT ---
# LOFT is mirrored per (dataset, length) as its own repo rather than as configs of
# one repo, so each combination is a separate load_dataset call. `_load_loft`
# concatenates dev+test, so both splits must be cached.
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
