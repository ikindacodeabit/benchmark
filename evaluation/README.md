[![Hugging Face Leaderboard](https://img.shields.io/badge/🤗%20HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)

# Evaluation

We support evaluation for all the presses implemented in the library, on a variety of popular benchmarks.

The same repository also contains a Recursive Language Model baseline for
vanilla-vs-RLM long-context comparisons. It shares this evaluator's benchmark
registry, dataset loaders, scorers, and result artifact names while retaining an
independent inference path. See the [RLM benchmark guide](rlm/README.md), and
[RLM.md](../RLM.md) for how the two backends relate and how RLM sub-call chunk sizes are
derived from a KV memory budget.

### Quick Start 🚀
> Evaluation requires some additional packages. You can install them with `uv sync --extra eval`
> Dataset caching and generation dependencies are documented in
> [benchmarks/README.md](benchmarks/README.md).

Running evaluation is straightforward! Make sure you are in the `evaluation` directory, then:

1. **Configure your evaluation** - Edit the global `evaluate_config.yaml`.
2. **Run the evaluation** - Execute the script: ```python evaluate.py```

Model-running commands read `evaluate_config.yaml`. The active default is the
non-quantized Synthetic-KV 32K native-context baseline. Local reference
templates can be kept in `../benchmark_artifacts/yml/`, which is intentionally
ignored by Git.
If you want, you can override the settings via command line, for instance:

```bash
python evaluate.py --dataset loogle --data_dir shortdep_qa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name expected_attention --compression_ratio 0.5
```

💡 Results (predictions and metrics) are automatically saved to the configured
`output_dir`. Repository configurations keep every generated run under the
single `benchmark_artifacts/results/` tree, grouped by dataset and task/context.
The complete `benchmark_artifacts/` directory is ignored by Git.


### Configuration 

Use `evaluate_config.yaml` as the single, self-contained global entry point.
Change settings such as the dataset fraction or model arguments directly in
this file. The ignored `../benchmark_artifacts/yml/` directory is only for
local reference copies.

Memory-budget matrices use one shared runner and one constants module. After
setting the matching dataset in `evaluate_config.yaml`, select a profile:

```bash
python run_matrix.py --profile synthetic-kv-64k
```

Profiles for LOFT, RULER, and Synthetic-KV are defined in
`matrix_constants.py`. The same runner supports full matrices,
`--baseline-only`, one `--memory-budget VALUE UNIT`, or one
`--configuration-id` for Slurm arrays. Distributed jobs can additionally pass
`--worker-id` and `--worker-count`.

Dataset loading and benchmark-specific schema normalization are centralized in
`benchmarks/loaders.py`. The evaluation runner calls its single dispatcher,
then applies only shared sampling and model-dependent preparation.

💡 Set `query_aware: true` to include the question in the context during compression. This enables query-aware compression as used in methods like SnapKV and FinchPress.


### Available Presses and Datasets 
We support evaluation with all the presses implemented in the library (and possible combinations). 

- All implemented presses are listed in the `PRESS_REGISTRY` variable in `evaluate_registry.py`.
- All implemented dataset are listed in `DATASET_REGISTRY` variable in `evaluate_registry.py`. 

At the moment, we support the following standard popular benchmarks:

- [Loogle](benchmarks/loogle/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/loogle))
- [RULER](benchmarks/ruler/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/ruler))
- [Zero Scrolls](benchmarks/zero_scrolls/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/zero_scrolls))
- [Infinitebench](benchmarks/infinite_bench/README.md) ([hf link](https://huggingface.co/datasets/MaxJeblick/InfiniteBench))
- [longbench](benchmarks/longbench/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench))
- [longbench-v2](benchmarks/longbenchv2/README.md)([hf link](https://huggingface.co/datasets/simonjegou/LongBench-v2))
- [Needle in a Haystack](benchmarks/needle_in_haystack/README.md)([hf link][Paul Graham's essays](https://huggingface.co/datasets/alessiodevoto/paul_graham_essays))
- [Synthetic 32K](benchmarks/synthetickv32k/README.md)
- [Synthetic-KV 64K](benchmarks/synthetickv64k/README.md)

The long-context dataset directories are self-contained:

```bash
ruler32k/ or ruler64k/
├── README.md
├── calculate_metrics.py
└── prepare_dataset.py

synthetickv32k/ or synthetickv64k/
├── README.md
├── calculate_metrics.py
├── generate_dataset.py
└── prepare_dataset.py
```

Each compact Synthetic-KV dataset contains:
  - `context`: ... 
  - `question`: ...
  - `answer_prefix`: ...
  - `answer`:  ...
  - `max_new_tokens`:  ...
`calculate_metrics.py` calculates metrics from the output of `evaluate.py`.


### Multi GPU Evaluation
Use the provided `evaluate.sh` script to run multiple presses simultaneously across different GPUs with varying compression ratios.

### Leaderboard 🥇
After evaluating your model, you can easily submit it to the [KVPress Leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard) on Hugging Face! Just copy the output directory in the huggingface space, and your method will soon be displayed in the leaderboard.
