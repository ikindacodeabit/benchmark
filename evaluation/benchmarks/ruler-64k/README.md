# RULER 64K benchmark

This benchmark uses `ollamaweights/Ruler-64k`, configuration `65536`, split
`test`. The cached dataset contains 6,500 examples: 500 examples for each of
the 13 RULER tasks.

The full matrix configuration samples 50% of each task with deterministic seed
42: 250 examples per task and 3,250 examples total. Result directories include
`fraction0.500` so they do not collide with full-dataset results.

## Offline dataset preparation

Run this once on the login node while internet access is available:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate kvpress
export HF_HOME="$HOME/.cache/huggingface"
cd "$HOME/kvpress"
python evaluation/benchmarks/ruler-64k/create_huggingface_data_for_offline.py
```

The compute-node job uses the same `HF_HOME` with Hugging Face offline mode.

## Evaluation

The dedicated configuration is
`benchmark_artifacts/yml/ruler/64k/all_tasks/evaluate_ruler64k_config.yaml`.
Each array task evaluates one RULER task six times:

1. KVzip reference run with compression ratio `0.01`.
2. KVzip run with a 512 MB logical KV budget.
3. KVzip run with a 1 GB logical KV budget.
4. KVzip run with a 2 GB logical KV budget.
5. KVzip run with a 3 GB logical KV budget.
6. KVzip run with a 4 GB logical KV budget.

Submit five workers. Together they automatically cover all 13 tasks:

```bash
cd "$HOME"
sbatch --array=0-4 ruler64k-job.sh
```

No later submission is required. Workers 0-2 process three tasks each and
workers 3-4 process two tasks each. Within every task, the reference and five
memory-budget configurations run sequentially. Each worker loads Qwen once and
reuses that model for all of its assigned tasks and budgets. Dataset text is
loaded once per task, while every configuration receives a fresh dataframe,
KVzip state, and KV cache.

Logs are kept under `kvpress/benchmark_artifacts/logs/`. Results are written
under `kvpress/benchmark_artifacts/results/ruler/64k/`.

The reported retained KV size is a logical/equivalent KVzip size. This KVPress
implementation uses head-specific masking and does not physically reduce the
allocated CUDA KV-cache tensor to the configured budget.

## L40 smoke test

Run one sample from only the first worker's first task (`cwe`) across the
reference and five budgets:

```bash
cd "$HOME"
KV_PRESS_MAX_TASKS_PER_WORKER=1 KV_PRESS_FRACTION=0.002 \
  sbatch --partition=l40 --qos=l40 --array=0 ruler64k-job.sh
```

The `fraction0.002` result-directory suffix keeps smoke-test results separate
from the full benchmark.
