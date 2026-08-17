# Direct Native Baseline Evaluator

This is an inference-independent baseline runner for every dataset registered
in this KVPress checkout. It uses the same benchmark dataset preparation and
canonical scorers so results remain comparable, but inference is performed by
ordinary Hugging Face `model.generate()` over each complete context/question.

It deliberately has no:

- `kvpress` package import
- press construction
- forward-hook registration
- KV reconstruction or eviction
- custom cache or model adapter
- compression ratio or memory budget

The only shared code is dataset loading, needle preparation, and metric
calculation under `evaluation/benchmarks/`. Model execution is independent and
uses the model's native Hugging Face cache.

## Supported datasets

The `dataset` field accepts every current registry entry:

`loogle`, `ruler`, `zero_scrolls`, `infinitebench`, `longbench`,
`longbench-e`, `longbench-v2`, `needle_in_haystack`, `aime25`, `math500`,
`loft`, `ruler32k`, `ruler64k`, `synthetic_kv`, and `synthetic_kv_32k`.

Use `data_dir` for the dataset task/configuration. It may be one string, a list
of strings, or `null` for datasets without a task configuration.

## Supported models

Normal text models load through `AutoModelForCausalLM`. Qwen3.5 loads through
its conditional-generation class with text-only inputs. Pre-quantized models
such as GPTQ are detected from their checkpoint configuration and placed on
the requested device during loading; they are not quantized again.

## Qwen3.5-27B GPTQ-Int4

Download on the login node if the local model directory is absent:

```bash
/home/rethinkingai-self/25m0820/miniconda3/envs/kvpress/bin/hf download \
  Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --local-dir /home/rethinkingai-self/25m0820/kvpress/Qwen3.5-27B-GPTQ-Int4
```

Submit the one-row LOFT NQ-32K smoke test to a DGX node:

```bash
cd /home/rethinkingai-self/25m0820/kvpress
/opt/slurm/bin/sbatch direct_baseline/slurm/qwen35_27b_gptq_int4_loft32k_smoke_dgx.sh
```

After the smoke test, the complete LOFT jobs are ready as:

```bash
/opt/slurm/bin/sbatch direct_baseline/slurm/qwen35_27b_gptq_int4_loft32k_dgx.sh
/opt/slurm/bin/sbatch direct_baseline/slurm/qwen35_27b_gptq_int4_loft128k_dgx.sh
```

For another benchmark, copy `configs/example_all_datasets.yaml` and select the
dataset/tasks. Do not add compression settings: the runner rejects press and
memory-budget keys.

Each task writes resumable `predictions.jsonl`, `predictions.csv`,
`metrics.json`, `run_summary.json`, and the resolved `config.json`.

## Independence check

This check is CPU-only and performs no model or dataset loading:

```bash
python direct_baseline/verify_independence.py
```
