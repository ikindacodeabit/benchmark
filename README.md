# KVPress + RLM long-context benchmarking

A fork of [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) (KV-cache compression
for 🤗 transformers) extended with a Recursive Language Model (RLM) harness, so the
two approaches to long context can be measured against each other on the same
datasets, with the same scorers, on one cost axis.

All inference is **local**: an OpenAI-compatible server (vLLM) on the GPU host for
the RLM arms, and an in-process HF pipeline for the KVPress arms. There is no
hosted-API path.

[**Open the benchmark results dashboard**](https://effortless-cupcake-470e7a.netlify.app/)
— completed dataset benchmarks, memory-budget sweeps, model variants, metrics, and
downloadable prediction files.

## Layout

| Path | What it is |
|---|---|
| `kvpress/` | The compression library. Presses live in `kvpress/presses/`; `KVPressTextGenerationPipeline` in `kvpress/pipeline.py` is the user-facing API. |
| `evaluation/` | Benchmark harness: datasets, canonical scorers, and the KVPress runner (`evaluate.py`, `run_matrix.py`). |
| `evaluation/rlm/` | The RLM harness — REPL scaffold, sub-call backends, and its runner. See `evaluation/rlm/README.md`. |
| `direct_baseline/` | Uncompressed HF baseline (`model.generate()`), deliberately importing no kvpress code. |
| `tests/` | Unit tests for presses, the pipeline, and the evaluation harness. |

## Running things

```bash
uv sync --extra eval && source .venv/bin/activate

# KVPress: one press/ratio sweep locally
cd evaluation && ./evaluate.sh

# KVPress: a memory-budget matrix (SLURM array job)
sbatch run-eval.sh yml/<config>.yaml <profile> <worker-count>

# RLM: serve a model, then run both arms against it
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
python -m evaluation.rlm.run_benchmark --dataset ruler32k --data-dir niah_single_1 --limit 5

# Join both backends into one comparison table
python -m evaluation.compare --dataset ruler32k
```

Every run — either backend — writes the same four artifacts to its own directory:
`predictions.csv`, `metrics.json`, `config.yaml`, `README.md`. That shared contract
is what makes `compare.py` able to put the two side by side.

## Further reading

- [**RLM.md**](RLM.md) — how RLM and kvpress coexist, and how sub-call chunk sizes are
  derived from a KV memory budget.
- [**evaluation/rlm/README.md**](evaluation/rlm/README.md) — running the RLM arms,
  the guards, and what makes the two backends comparable.
- [**KNOWN_ISSUES.md**](KNOWN_ISSUES.md) — expected test failures, deliberate design
  limitations, and upstream bugs left in place. Read this before filing anything.
- [**AGENTS.md**](AGENTS.md) — contribution conventions (SPDX headers, DCO, formatting).
