# RLM Long-Context Benchmarking

This directory contains the Recursive Language Model (RLM) harness within the
KVPress evaluation repository. RLM and KVPress keep independent inference paths,
but share the benchmark registry, Hugging Face loaders, canonical scorers, and
the `predictions.csv` / `metrics.json` / `config.yaml` result contract. Run all
commands below from the repository root.

## Goal
Quantify how Recursive Language Models (RLMs) compare against vanilla long-context
inference on standard long-context benchmarks, using NVIDIA NIM hosted APIs as the
inference backend and Prajna for orchestration, data prep, and scoring.

## Architecture decision
**Phase 1 (this repo): NIM cloud API.** All model calls go to
`https://integrate.api.nvidia.com/v1` (OpenAI-compatible). Prajna runs the RLM
scaffold (REPL loop, recursion, scoring) — no GPU needed for this phase, so submit
to a CPU-friendly partition and save your GPU quota.

**Phase 2 (later): self-hosted inference on Prajna GPUs.** Once the harness works,
swap `base_url` to a local vLLM/SGLang server running on the DGX A100 / L40S nodes
(e.g. Qwen3-8B to reproduce the paper's RLM-Qwen3-8B setting). The code is backend
agnostic — only the config changes.

## Pipeline
```
[login node]                          [compute node, CPU partition]
download datasets (HF) ──> /scratch ──> SLURM array job: run_benchmark.py
                                          ├── condition A: vanilla (full context in prompt)
                                          ├── condition B: RLM (context as REPL variable)
                                          └── per-example JSONL results + resumable checkpoints
                                       ──> score.py ──> results table
```

## Steps
1. **Account/API setup**
   - Get an NVIDIA API key at build.nvidia.com (starts with `nvapi-`).
   - On Prajna: `echo 'export NVIDIA_API_KEY=nvapi-...' >> ~/.bashrc` (never hardcode in scripts).
2. **Environment** (login node is fine for this):
   - `uv sync --extra eval`
   - `source .venv/bin/activate`
3. **Connectivity check** — HPC compute nodes often have no outbound internet.
   Run `sbatch benchmark_artifacts/slurm_jobs/rlm/setup/check_net.slurm`. If it fails, ask hpc@iitb.ac.in about an
   HTTP proxy for compute nodes, or run the harness on the login node ONLY for
   tiny smoke tests (API orchestration is light, but per Prajna policy real runs
   must not live on login nodes).
4. **Data prep** (login node, since it needs internet):
   - `bash benchmark_artifacts/slurm_jobs/rlm/setup/download_data.sh` — caches the shared LongBench-v2 and RULER-32K datasets under `$HF_HOME`
   - Synthetic NIAH/RULER-style tasks are generated locally (no download needed).
5. **Smoke test** (~5 examples): `python -m evaluation.rlm.run_benchmark --dataset niah --limit 5 --mode both`
   For a shared RULER-32K subset, use:
   `python -m evaluation.rlm.run_benchmark --dataset ruler32k --data-dir niah_single_1 --limit 5 --mode both`.
6. **Full runs**: `sbatch benchmark_artifacts/slurm_jobs/rlm/all_tasks/run_eval.slurm`.
7. **Score & compare**: `python -m evaluation.rlm.score benchmark_artifacts/results/rlm`

Each RLM run gets its own directory containing a resumable `checkpoint.jsonl`,
the three common result artifacts, and (for RLM mode) per-example transcripts.
The canonical benchmark scorer—not the harness's quick progress-match flag—is
the authoritative score in `metrics.json`.

## Comparing against KVPress
Run both backends on the same `dataset` / `data_dir` / `model`, then join them:

```bash
python -m evaluation.compare --dataset ruler32k --csv comparison.csv
```

`compare.py` walks `evaluation/results/`, reads each run's `config.yaml` and
`metrics.json`, and prints one row per run. What makes the rows commensurable:

- **Same scorer.** Both backends call `benchmarks/results.score_prediction_frame`,
  so `score` is literally the same function applied to the same gold answers.
- **Same cost axis.** `context_tokens_held` is KVPress's
  `average_retained_context_tokens` and RLM's `average_peak_context_tokens` —
  both answer "how much context was live at once", which is what KV memory is
  proportional to. Plot score against this column to put a compression sweep and
  a `--max-context-tokens` sweep on one chart.

Two asymmetries the harness records rather than hides, because ignoring either
one silently flatters RLM:

- **Errors.** An RLM example that dies on an API error is dropped from the score
  and counted in `runtime.errors`. KVPress runs locally and cannot fail this
  way, so scoring an outage as a zero would understate RLM for a reason that has
  nothing to do with the method. Treat a run with many errors as incomplete, not
  as a result — check the `errors` column before believing a score.
- **Truncation.** KVPress compresses the full context; vanilla truncates it. The
  `context_retained` column is the fraction that survived, so a low vanilla score
  can be read as a property of the char/token limit rather than of the model.
  Set `--vanilla-max-prompt-tokens` to `(max-model-len - max-tokens - margin)`;
  without it, densely-tokenising subsets (RULER `cwe`, `niah_multikey_3`)
  overflow the served window and score 0.0 from a harness error.

## Runaway guards and the scratchpad
All four are toggleable and independent; pass `0` to disable one without
affecting the others. Defaults are on because an unguarded run can burn a whole
SLURM allocation on one pathological example, but a debugging session that wants
to watch a single example run to completion can switch any of them off.

| Flag | Default | Off | What it bounds |
|---|---|---|---|
| `--exec-timeout` | `60` s | `0` | Pure-Python time for ONE code block, so a model-generated infinite loop can't hang the sweep. Time inside `llm_query` is excluded — the watchdog is paused around sub-calls, so a slow-but-legitimate API call is never mistaken for a runaway loop. |
| `--run-timeout` | `900` s | `0` | Wall-clock for ONE example, ending it with `end_reason=run_timeout`. Needed because `--exec-timeout` deliberately does not bound `llm_query` time. |
| `--max-sub-calls` | `40` | `0` | `llm_query` calls per example. Past the cap, calls return a notice instead of hitting the API, so the model degrades gracefully rather than erroring. |
| `--scratchpad` | off | (omit) | Opt-in `note(text)` REPL tool. Notes are re-shown every turn and survive budget eviction, giving the model a durable place to keep findings. Size it with `--max-notes-tokens`. |

`--exec-timeout` relies on `SIGALRM`, so it is a no-op off the main thread or on
platforms without it; it also cannot interrupt a C-level regex. `--run-timeout`
has neither limitation and is the backstop.

Because `--scratchpad` and `--max-context-tokens` both change results, both
appear in the run directory name — otherwise two configurations would share a
`checkpoint.jsonl` and silently merge. Nothing parses that name; `compare.py`
reads `config.yaml`.

The root and sub clients now share one `RateLimiter`, so `--rpm` is a per-account
cap. Previously each client had its own and the process could issue ~2x `--rpm`
and trip 429s.

## Benchmarks (in order of effort)
| Benchmark | Why | Source |
|---|---|---|
| Synthetic NIAH / RULER-style | Sanity check; both vanilla & RLM should ace it | generated locally |
| LongBench v2 | Standard, broad long-context QA | HF: `THUDM/LongBench-v2` |
| OOLONG | The benchmark where the RLM paper showed the biggest gap | HF (see download script) |
| ∞Bench / BrowseComp-Plus | 100k–10M token stress tests (RLM-only territory) | HF |

## Experimental conditions
- **Vanilla**: full context stuffed into the prompt of a long-context NIM model
  (truncate at model limit; record truncation).
- **RLM**: same base model as the REPL "root", context held as a Python variable,
  `llm_query()` available for recursive sub-calls (depth 1 to start).
- Hold the base model constant across both conditions. Suggested starters from the
  NIM catalog: a large-context model for the vanilla ceiling and a cheaper small
  model to replicate the paper's "small RLM beats big vanilla" claim.

## Metrics
- Accuracy / F1 per task (exact-match for NIAH, substring/F1 for QA)
- Tokens used per query (prompt + completion, summed over all recursive calls)
- Wall-clock latency per query
- REPL steps used; failure modes (max-steps exhausted, code errors, API errors)

## Practical constraints to engineer around
- NIM free tier: ~40 req/min and limited credits → built-in rate limiter,
  exponential backoff, and JSONL checkpointing (resume with the same command).
- Vanilla condition burns credits fast at 100k+ tokens; start with `--limit 50`.
- Set `--time` in SLURM generously: RLM runs are many sequential API calls.
