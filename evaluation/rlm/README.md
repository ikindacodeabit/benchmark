# RLM Long-Context Benchmarking

This directory contains the Recursive Language Model (RLM) harness within the
KVPress evaluation repository. RLM and KVPress keep independent inference paths,
but share the benchmark registry, Hugging Face loaders, canonical scorers, and
the `predictions.csv` / `metrics.json` / `config.yaml` result contract. Run all
commands below from the repository root.

## Goal
Quantify how Recursive Language Models (RLMs) compare against vanilla long-context
inference on standard long-context benchmarks, with all inference served locally:
a vLLM (or any OpenAI-compatible) server on the GPU host for the root and vanilla
arms, and optionally an in-process KVzip-compressed sub model (`--sub-backend kvzip`).

## Architecture
The harness talks to whatever `--base-url` points at (default
`http://localhost:8000/v1`). The two supported deployment shapes:

- **Bare-metal GPU host** (infolab-style, no scheduler):
  `evaluation/rlm/loft128k/run_infolab.sh` launches one vLLM server per GPU and
  drives `run_cells.sh`, which runs the benchmark arms against it.
- **SLURM**: `evaluation/rlm/loft128k/slurm/loft128k_a100.slurm` is an array job
  that starts one vLLM server per dataset on its allocated GPU and runs the same
  cells. `evaluation/rlm/slurm/download_data.sh` prefetches HF datasets on a
  login node for air-gapped compute nodes.

## Pipeline
```
[login node]                          [GPU node]
download datasets (HF) ──> $HF_HOME ──> vllm serve <model> --port 8000
                                        ──> run_benchmark.py --base-url http://localhost:8000/v1
                                              ├── condition A: vanilla (full context in prompt)
                                              ├── condition B: RLM (context as REPL variable)
                                              └── per-example JSONL results + resumable checkpoints
                                        ──> compare.py ──> results table
```

## Steps
1. **Environment**: `uv sync --extra eval`, then `source .venv/bin/activate`.
2. **Data prep** (login node if compute nodes lack internet):
   `bash evaluation/rlm/slurm/download_data.sh` — caches the shared datasets
   under `$HF_HOME`. Synthetic NIAH/RULER-style tasks are generated locally.
3. **Serve a model**: `vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000`
   (or use `run_infolab.sh`, which does this per GPU with the right flags).
4. **Smoke test** (~5 examples):
   `python -m evaluation.rlm.run_benchmark --dataset niah --limit 5 --mode both`
   For a shared RULER-32K subset:
   `python -m evaluation.rlm.run_benchmark --dataset ruler32k --data-dir niah_single_1 --limit 5 --mode both`.
5. **Full runs**: `bash evaluation/rlm/loft128k/run_infolab.sh` (bare metal) or
   `sbatch evaluation/rlm/loft128k/slurm/loft128k_a100.slurm` (SLURM).
6. **Score & compare**: `python -m evaluation.compare --backend rlm`

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

For an arm with KV-compressed sub-calls, `context_tokens_held` is
`max(root_peak, sub_peak)`, with the root-only figure kept beside it as
`context_tokens_root`. Counting only the root would flatter that arm badly: while
the root holds ~2k tokens, the sub model holds a whole slice of KV on the GPU,
and `KVzipPress` masks it rather than freeing it.

Three asymmetries the harness records rather than hides, because ignoring any of
them silently flatters RLM:

- **Errors.** An RLM example that dies on a server/harness error is dropped from
  the score and counted in `runtime.errors`. KVPress runs in-process and cannot
  fail this way, so scoring an outage as a zero would understate RLM for a reason
  that has nothing to do with the method. Treat a run with many errors as
  incomplete, not as a result — check the `errors` column before believing a score.
- **Truncation.** KVPress compresses the full context; vanilla truncates it. The
  `context_retained` column is the fraction that survived, so a low vanilla score
  can be read as a property of the char/token limit rather than of the model.
  Set `--vanilla-max-prompt-tokens` to `(max-model-len - max-tokens - margin)`;
  without it, densely-tokenising subsets (RULER `cwe`, `niah_multikey_3`)
  overflow the served window and score 0.0 from a harness error.
- **Abstentions.** An RLM example can end with `FINAL_NONE` — the model searched
  and reports finding nothing. That scores 0 exactly like a wrong answer, so
  `runtime.abstained` is reported separately: an arm that abstains honestly and
  one that hallucinates confidently are not the same result, and the score column
  alone cannot tell them apart. Vanilla has no equivalent ending.
- **Retrieval.** A `--search-k > 0` arm has an information channel the others do
  not: a BM25 index over the whole document. Vanilla truncates, KVPress
  compresses, and this arm *selects*. The only fair comparator is a `--search-k 0`
  run of the same commit, which is why the flag is stamped into the run directory
  and `config.yaml` rather than becoming a default.
  `runtime.gold_in_retrieved_fraction` is a loose answer-string presence
  diagnostic among scored examples with recorded retrieved spans. It is not a
  score ceiling: the root can read elsewhere, and a matching string does not
  establish that the retrieved text supplies sufficient evidence to answer.
  Two caveats travel with it. The treatment is not one variable: the primitive,
  the worked example, the locate bullet and the abstention gate all move together.
  And `document_coverage_fraction` changed meaning — spans are now resolved before
  the dense fold, so it was structurally `0.0` on every earlier `--sub-backend
  http` run. `config.yaml` records `coverage_attribution: spans_v2`; do not
  compare a coverage number across that line.

Search-enabled RLM runs with `--search-overlap 0` now record
`search_index_revision: 2`: an earlier cleanup loop incorrectly discarded all
but the first window. These runs require a fresh results directory when an
existing configuration lacks that revision. Default overlapping windows are
unchanged.

## Runaway guards and the scratchpad
All four are toggleable and independent; pass `0` to disable one without
affecting the others. Defaults are on because an unguarded run can burn a whole
SLURM allocation on one pathological example, but a debugging session that wants
to watch a single example run to completion can switch any of them off.

| Flag | Default | Off | What it bounds |
|---|---|---|---|
| `--exec-timeout` | `60` s | `0` | Pure-Python time for ONE code block, so a model-generated infinite loop can't hang the sweep. Time inside `llm_query` is excluded — the watchdog is paused around sub-calls, so a slow-but-legitimate server call is never mistaken for a runaway loop. |
| `--run-timeout` | `900` s | `0` | Wall-clock for ONE example, ending it with `end_reason=run_timeout`. Needed because `--exec-timeout` deliberately does not bound `llm_query` time. |
| `--max-sub-calls` | `40` | `0` | `llm_query` calls per example. Past the cap, calls return a notice instead of hitting the server, so the model degrades gracefully rather than erroring. |
| `--scratchpad` | off | (omit) | Opt-in `note(text)` REPL tool. Notes are re-shown every turn and survive budget eviction, giving the model a durable place to keep findings. Size it with `--max-notes-tokens`. |
| `--search-k` | `0` (off) | `0` | Opt-in `search(query)` REPL tool: BM25 over fixed overlapping windows of the document, returning this many ranked windows. Pass them to `llm_query(question, hits)` so the payload stays attributable. The model may ask for fewer windows but never more, so a worked example cannot override a swept grid cell. Size the windows with `--search-window` / `--search-overlap`. |

`--exec-timeout` relies on `SIGALRM`, so it is a no-op off the main thread or on
platforms without it; it also cannot interrupt a C-level regex. `--run-timeout`
has neither limitation and is the backstop.

Because `--scratchpad` and `--max-context-tokens` both change results, both
appear in the run directory name — otherwise two configurations would share a
`checkpoint.jsonl` and silently merge. Nothing parses that name; `compare.py`
reads `config.yaml`.

The name cannot carry *every* result-affecting knob without becoming unreadable,
so the rest (`--limit`, `--split`, `--max-steps`, the timeouts, the sub model,
…) are enforced on **resume** instead: `config.yaml` is written before the first
example runs, and a later run whose settings disagree with it aborts rather than
appending into someone else's checkpoint.

`--split dev|test|all` picks a split where the dataset has them. It matters most
for LOFT, which ships 10 dev rows followed by 100 test rows: with `--split all`
(the default) a `--limit 10` smoke run silently evaluates *only* dev, and lands
in the results tree looking like a real run that scored 0.000. Use `--split dev`
for smoke runs — it also gets its own directory — and `--split test` for real ones.

## Benchmarks (in order of effort)
| Benchmark | Why | Source |
|---|---|---|
| Synthetic NIAH / RULER-style | Sanity check; both vanilla & RLM should ace it | generated locally |
| LongBench v2 | Standard, broad long-context QA | HF: `THUDM/LongBench-v2` |
| OOLONG | The benchmark where the RLM paper showed the biggest gap | HF (see download script) |
| ∞Bench / BrowseComp-Plus | 100k–10M token stress tests (RLM-only territory) | HF |

## Experimental conditions
- **Vanilla**: full context stuffed into the prompt of the served long-context
  model (truncate at model limit; record truncation).
- **RLM**: same base model as the REPL "root", context held as a Python variable,
  `llm_query()` available for recursive sub-calls (depth 1 to start).
- Hold the base model constant across both conditions.

## Metrics
- Accuracy / F1 per task (exact-match for NIAH, substring/F1 for QA)
- Tokens used per query (prompt + completion, summed over all recursive calls)
- Wall-clock latency per query
- REPL steps used; failure modes (max-steps exhausted, code errors, server errors)

## Practical constraints to engineer around
- Vanilla at 100k+ tokens is slow even served locally; start with `--limit 50`.
- Set `--time` in SLURM generously: RLM runs are many sequential server calls.
- JSONL checkpointing means an interrupted run resumes with the same command.
