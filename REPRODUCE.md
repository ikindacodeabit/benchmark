# Reproducing the RLM + KVzip runs on LOFT

This is the from-scratch checklist for running the RLM arm with KV-compressed
sub-calls (`--sub-backend kvzip`) against the LOFT RAG datasets, and for the
`longbench128k` fixed-chunk grid built on top of it. It exists because a plain
`git clone` + `uv sync` is **not** enough: the run needs an NVIDIA GPU, a served
root model, a specific `transformers` pin, and network access to third-party
Hugging Face datasets.

For design context read [`RLM.md`](RLM.md), [`evaluation/rlm/README.md`](evaluation/rlm/README.md),
and [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) (the last one lists the ~42 CPU-only test
failures that are expected and not caused by this work).

## 0. Branch

The RLM/KVzip + fixed-chunk-grid work lives on **`feature/fixed-chunk-grid`**
until it is merged. `master` does not have it.

```bash
git clone <this-fork> benchmark && cd benchmark
git checkout feature/fixed-chunk-grid
```

## 1. Hardware / OS

- One CUDA GPU. The default recipe serves `Qwen/Qwen3-4B-Instruct-2507` with
  vLLM **and** loads a second copy of the model in-process for the sub-calls.
  Budget for both: ~16 GB for the served root plus `--sub-min-free-gib` (default
  14 GiB) free for the sub model.
- `kvpress`'s `KVzipPress` is **logical** compression — evicted keys are masked,
  not freed — so a sub-call still needs its *full uncompressed* KV resident.
  `--memory-budget` controls how many tokens are logically retained, not GPU
  memory saved. Size `--sub-min-free-gib` for the uncompressed context.
- No CPU-only path for the actual benchmark. The unit tests run CPU-only; the
  run does not.

## 2. Python environment

Use **Python 3.12**. The committed `.venv/` is 3.13, where `fire` fails to import
(`ModuleNotFoundError: No module named 'pipes'`) and pytest cannot even collect
the suite. `.venv-mac/` in this checkout is a 3.12 CPU test env; make an
equivalent GPU one.

```bash
uv sync --extra eval          # or: python3.12 -m venv .venv && pip install -e ".[eval]"
source .venv/bin/activate

# REQUIRED, and must come LAST. `.[eval]` pulls transformers >= 4.56; vLLM 0.8.5
# calls Qwen2Tokenizer.all_special_tokens_extended, which transformers 5.x
# removed, so `vllm serve` dies ~5 min into startup with a tokenizer AttributeError
# that looks like a GPU/readiness problem. Pin to the version vLLM expects:
pip install "transformers==4.51.3"
```

pip will warn that this conflicts with the `kvpress` presses' declared range.
That warning is expected and harmless here: the KVzip sub-call path in this arm
uses `kvpress.presses.kvzip_press.KVzipPress` + `KVPressTextGenerationPipeline`
from **this repo**, which work on 4.51.3. (Adjust the exact pin if your vLLM
version differs — the rule is "match vLLM's transformers requirement".)

Extras needed for the LongBench scorer that `longbench128k` reuses: `jieba`,
`fuzzywuzzy`, `rouge` (all in `.[eval]`; install by hand if you slimmed the
environment).

## 3. Data

### LOFT

`_load_loft` calls `load_dataset("f20180301/loft-rag-<dataset>-<length>")`
directly from the Hub — a **third-party** namespace this project does not own.
You need network access at run time (or a warmed `$HF_HOME`), and if that
namespace is ever gated you are blocked with no local fallback.

Available subsets (`--data-dir`): `{nq, hotpotqa, musique, qampari, quest}` ×
`{32k, 128k, 1m}`, e.g. `nq_128k`. Only the RAG task family is mirrored;
retrieval / ICL / SQL LOFT tasks have no loader here.

Pre-warm the cache (useful for air-gapped compute nodes):

```bash
bash evaluation/rlm/slurm/download_data.sh
```

### longbench128k (the fixed-chunk grid)

Built locally from `$RLM_DATA_DIR/longbench.jsonl` (default `~/rlm_data`), the
standard LongBench dump. Nothing is fetched from the Hub.

```bash
python -m evaluation.benchmarks.longbench128k.build_dataset --per-subset 60
# writes $RLM_DATA_DIR/longbench128k/<subset>/{data.parquet,build_manifest.json}
```

## 4. Serve the root model

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

Any OpenAI-compatible server works; point `--base-url` at it. The harness never
starts or stops the server.

## 5. Run

### Smoke (no compression, 5 dev rows)

```bash
python -m evaluation.rlm.run_benchmark \
  --dataset loft --data-dir nq_128k --split dev --limit 5 \
  --mode rlm --base-url http://localhost:8000/v1
```

### RLM + KVzip on LOFT

```bash
python -m evaluation.rlm.run_benchmark \
  --dataset loft --data-dir nq_128k --split test --limit 50 \
  --mode rlm --scratchpad --max-steps 50 \
  --base-url http://localhost:8000/v1 \
  --root-model Qwen/Qwen3-4B-Instruct-2507 \
  --sub-model  Qwen/Qwen3-4B-Instruct-2507 \
  --sub-backend kvzip --press kvzip \
  --memory-budget 1 --memory-budget-unit GB \
  --max-subcall-chars auto --target-compression-ratio 0.5 \
  --sub-max-tokens 128 --sub-min-free-gib 14 \
  --exec-timeout 60 --run-timeout 3600 \
  --out evaluation/results
```

Notes:

- `--press no_press` keeps the identical load/generate path with no pruning — the
  control arm.
- `--max-subcall-chars auto` requires `--sub-backend kvzip` **and** one of
  `--target-compression-ratio` / `--compression-factor`.
- With `auto` + a target ratio, the chunk can be capped by `--sub-max-context-tokens`
  (default 34000) or GPU fit rather than by the KV budget, in which case the
  realized ratio differs from the target. Check `runtime.subcall_sizing_binding`
  and `runtime.average_sub_compression_ratio` in `metrics.json`; a `!!` line is
  printed at startup when the target will not be delivered.
- Auto sizing reads live GPU free memory, so it is **not** reproducible across
  hosts. The resolved size is saved as `max_subcall_chars_resolved` in
  `config.yaml`; a resume that would drift >2% aborts.

### Exact fixed-chunk grid (longbench128k)

`--fixed-chunk` makes the auto-sized chunk exact (oversized char floor, question
room reserved, every sub-call token-truncated to `budget × factor`, hard failure
if the model window or GPU would shrink the cell). `--compression-factor N` is
the human-readable axis (`ratio = 1 - 1/N`; `N=1` is uncompressed).

```bash
python -m evaluation.rlm.run_benchmark \
  --dataset longbench128k --data-dir hotpotqa --split test --limit 50 \
  --mode rlm --scratchpad --max-steps 50 \
  --base-url http://localhost:8000/v1 \
  --sub-backend kvzip --press kvzip \
  --memory-budget 8192 --memory-budget-unit tokens \
  --max-subcall-chars auto --compression-factor 8 --fixed-chunk \
  --sub-max-tokens 128 --sub-min-free-gib 14 \
  --out evaluation/results/fixedgrid
```

The full six-subset × budget × factor sweep, its locking, and the results
renderer are in [`evaluation/rlm/fixedgrid/README.md`](evaluation/rlm/fixedgrid/README.md)
(`run_grid.sh`, then `python evaluation/rlm/fixedgrid/grid_table.py --results ...`).

## 6. Score and compare

Every run writes `predictions.csv`, `metrics.json`, `config.yaml`, `README.md`
to its own directory. The canonical scorer in `metrics.json` — not the quick
`progress_match_loose` flag — is authoritative.

```bash
python -m evaluation.compare --dataset loft --backend rlm --csv comparison.csv
```

## 7. Tests

```bash
python -m pytest tests/evaluation/ -q       # RLM / kvzip / sizing / fixedgrid / longbench128k
```

These run CPU-only with fakes on Python 3.12 and do **not** touch LOFT data or a
real GPU. `python -m pytest tests/` reports ~42 failures on a CPU-only machine
(`test_decoding_compression.py`, `test_pipeline.py`, …) — all pre-existing and
documented in `KNOWN_ISSUES.md`.

## Known gaps for outside reproduction

- `evaluation/rlm/loft128k/run_infolab.sh`, `run_cells.sh`, `run-eval.sh` and the
  SLURM job are written for the IITB infolab hosts (hard-coded paths, GPU-free
  thresholds, conda env names). Treat them as references, not turnkey scripts.
  `slurm/loft128k_a100.slurm` is self-labelled UNVERIFIED.
- LOFT data availability depends on the external `f20180301` HF namespace staying
  public.
