# LOFT-128k: vanilla vs RLM vs RLM+scratchpad vs RLM+scratchpad+KV-compression

Experiment arms on LOFT's five RAG subsets at 128k context, all on
`Qwen/Qwen3-4B-Instruct-2507`:

| arm | invocation | what it tests |
|---|---|---|
| `vanilla` | `--mode both` | the whole 131k-token document in the context window |
| `rlm` | `--mode both` | document held in a REPL variable, root model chunks and recurses |
| `rlm+scratchpad` | `--mode rlm --scratchpad` | same, plus a persistent `note()` buffer that survives eviction |
| `rlm+scratchpad+press` (4a) | `... --sub-backend kvpress` | sub-calls read their slice through a KVzip-compressed KV cache, same 32k-char chunks as arms 2–3 (isolates the press effect) |
| `rlm+scratchpad+press` (4b) | `... --sub-backend kvpress --max-subcall-chars 131072` | same, but ~32k-TOKEN chunks — fewer, bigger reads, the regime compression is supposed to enable |
| `vanilla+press` (cell 5) | `run_infolab.sh kvzip-baseline` | the press alone, through the standard kvpress `evaluate.py` path |

`--mode both` produces the first two arms in one pass, so the vanilla arm is not
recomputed for the scratchpad comparison. Every arm lands in its own run
directory (the run-dir name carries `scratchpad` / press+ratio / `subN`
components), and each is resumable through its own `checkpoint.jsonl`.

## Why this model

| | |
|---|---|
| layers / KV heads / head_dim | 36 / 8 / 128 → **144 KiB per token** of bf16 KV |
| `max_position_embeddings` | **262144** |
| weights (bf16) | ~8 GB |
| one 128k sequence | ~18 GB of KV |

128k is comfortably inside the native window, so **no YaRN** — unlike the
Qwen3-8B LOFT-128k runs in `Loft-Qwen3-8B-README.md`, which needed YaRN-4 to
stretch a 32768 window and paid for it in quality.

It is also a **non-thinking** model. Do not pass `--no-think` and do not serve
with `--reasoning-parser`: it emits no `<think>` blocks, and the `enable_thinking`
chat-template kwarg does not apply to its template. The whole `{think, nothink}`
axis that the Qwen3-8B scripts carry is gone.

## Running it — infolab (bee/fox)

These hosts have full internet, so there is no pre-staging step: weights and LOFT
parquet download on first use.

They also have **no scheduler**, and the GPUs are shared and unreserved. Nothing
restarts a dead run and nothing stops another user claiming a card mid-setup, so
run everything under `tmux`.

**Clone onto scratch, not `$HOME`.** `$HOME` is a small NFS quota on these hosts;
the venv alone (torch + vLLM) is ~15 GB and the model weights another ~8 GB.

```bash
cd /mnt/nas/$USER
git clone -b rlm-scratchpad-loft128k https://github.com/Rahul-Chhabra-27/benchmark.git
cd benchmark

tmux new -s rlm
bash evaluation/rlm/loft128k/run_infolab.sh setup
bash evaluation/rlm/loft128k/run_infolab.sh auto
```

Use the `setup` subcommand rather than running `uv sync` by hand: it redirects the
pip/uv/HF caches to scratch *before* installing anything. Those caches default to
`$HOME` and are several GB, so the ordering is the whole point.

`auto` picks every card with ≥30 GB free (ranked by **utilisation**, not free
memory — an idle-looking card can be pegged at 99% SM by a small-footprint job),
starts one vLLM per card, and deals the five datasets round-robin across them. A
4B model needs no tensor parallelism; N independent servers is the right way to
use N cards.

Smoke first, cheaply, at 32k:

```bash
DATASETS="nq" LENGTH=32k LIMIT=3 SERVERS=1 bash evaluation/rlm/loft128k/run_infolab.sh auto
```

then repeat at `LENGTH=128k LIMIT=2` for the real memory check, then drop the
overrides for the full grid.

To drive the halves by hand instead, in two tmux windows:

```bash
GPU=1 bash evaluation/rlm/loft128k/run_infolab.sh serve
DATASETS="nq" bash evaluation/rlm/loft128k/run_infolab.sh run
```

**Contention is the main risk.** On an empty 48 GB card the util cap lands at 0.85
(~33 GB of KV, ~1.8 concurrent 128k sequences). On a card already holding another
user's 13 GB job it lands near 0.59 (~20 GB of KV, ~1.1 sequences) — still
correct, just slow. Check `nvidia-smi` before launching and prefer the emptiest
cards.

Each subset is 110 examples: `_load_loft` concatenates **dev (10) then test
(100)**, in that order. So `LIMIT` values of 10 or less sample the dev split only
— fine for a smoke test, not a result. `LIMIT=110` is the whole subset.

## The KV-compression arms (4a/4b and the kvzip baseline)

### Two venvs, one driver

The vLLM **server** needs `transformers==4.51.3` (5.x removed
`all_special_tokens_extended`); kvpress needs `transformers>=4.56` (the new
`Cache.layers` API). They cannot share an environment, so `setup` creates two:
`.venv` serves vLLM, `.venv-kvpress` runs everything press-related. The trick
that makes arm 4 work with no HTTP shim: the benchmark **driver** imports
neither vllm nor transformers on its own, so it runs from `.venv-kvpress`,
talks to the vLLM ROOT server over HTTP, and hosts the compressed SUB model
in-process (`evaluation/rlm/kvpress_backend.py`).

### What actually changes inside the RLM

With `--sub-backend kvpress` the REPL tool becomes
`llm_query(question, context_text)`: the slice rides through
`KVPressTextGenerationPipeline`, which compresses ONLY the slice's KV during
prefill; the question (and `SUB_SYSTEM_PROMPT`) stay uncompressed. Two
documented deviations from the NIM path:

1. the kvpress pipeline has no system-role support, so the sub system prompt is
   prepended to the question side (identical across 4a/4b/ratios, so internal
   comparisons stay valid);
2. **KVzip here is logical compression** — evicted keys are masked via the
   attention patch, not freed — so `retained`-token metrics are quality knobs,
   not memory savings. Budget GPU memory for the full uncompressed KV.

### Colocation budget (one 48 GB card, `KVPRESS_ARMS=1`)

The root never sees the document (only REPL transcripts), so its server shrinks
to make room for the in-process sub model:

| tenant | memory |
|---|---|
| vLLM root, `--max-model-len 65536`, capped at `ROOT_BUDGET_MIB` | ~21 GB (8 weights + overhead + KV pool) |
| HF sub model | ~8.1 GB weights |
| one 34k-token sub prefill (full KV — KVzip masks, doesn't free) | ~4.9 GB |
| KVzip scoring transient | ~2.5 GB |
| **total** | **~37 GB** (≈11 GB margin for co-tenants) |

A root transcript that outgrows 65536 tokens gets a 400, is recorded as a
harness error, and is retried on resume — raise `KV_ROOT_MAX_LEN` if it
recurs. `SUB_GPUS="i j"` instead deals dedicated sub cards to the lanes and the
servers keep the full 139264 window.

### Running

```bash
# one-example smoke, arm 4a then 4b, at 32k
KVPRESS_ARMS=1 KV_RATIOS=0.5 KV_ARMS=4a DATASETS=nq LENGTH=32k LIMIT=1 SERVERS=1 \
    bash evaluation/rlm/loft128k/run_infolab.sh auto
KVPRESS_ARMS=1 KV_RATIOS=0.5 KV_ARMS=4b DATASETS=nq LENGTH=32k LIMIT=1 SERVERS=1 \
    bash evaluation/rlm/loft128k/run_infolab.sh auto

# the real grid: 5 datasets x {0.5, 0.75} x {4a, 4b}
KVPRESS_ARMS=1 LENGTH=128k bash evaluation/rlm/loft128k/run_infolab.sh auto

# cell 5: press-only baseline through evaluate.py (smoke with FRACTION=0.02)
DATASETS=nq KV_RATIOS=0.5 FRACTION=0.02 bash evaluation/rlm/loft128k/run_infolab.sh kvzip-baseline
bash evaluation/rlm/loft128k/run_infolab.sh kvzip-baseline
```

`KVPRESS_ARMS=1` runs ONLY the arm-4 lanes. Deliberate: the colocated server has
a reduced window, and a vanilla arm pointed at it would shrink-retry and merge
truncated rows into the existing vanilla checkpoints.

KVzip pays 2–3 extra prefill passes for its scoring, so a 4b sub-call on a
32k-token slice runs about a minute; the lanes budget for it
(`--run-timeout 2400/3600`, `--sub-max-tokens 512`, `--max-sub-calls 40/16`).

### Reading the results

Two run-level numbers in `metrics.json → runtime` decide whether the arm
measured anything:

- **`sub_split_call_fraction`** — how often the root actually used the two-arg
  form. Near 0 means the arm degenerated to dense one-arg calls and is NOT
  measuring the press.
- **`average_sub_retained_context_tokens`** vs `average_sub_context_tokens` —
  the sub-side analogue of KVPress's `average_retained_context_tokens`; the
  ratio between them should track `--compression-ratio`.

### Prajna

`slurm/loft128k_a100.slurm` is a SLURM job array for Prajna's A100s, kept for if
access is restored. It has **never been submitted** — Prajna was unavailable when
this was written. Verify the partition with `sinfo -s` and stage the cache on the
login node (`hf download Qwen/Qwen3-4B-Instruct-2507` plus
`bash evaluation/rlm/slurm/download_data.sh`, since compute nodes are air-gapped)
before trusting it.

## Verifying a run is valid

**Check `truncated_fraction` before believing any number.** In `metrics.json` for
the vanilla run dir it must be `0`.

Measured on `f20180301/loft-rag-nq-128k`: every context is **475,516 characters**
(identical across examples — LOFT shares one corpus per dataset and prompts
corpus-in-context), or roughly 119k tokens. `--vanilla-char-limit` defaults to
**400,000**, so the default silently drops ~16% off the end of every document and
hands the RLM arm a win it did not earn. `run_cells.sh` sets
`--vanilla-char-limit 700000` and bounds the prompt by tokens via
`--vanilla-max-prompt-tokens 134000`. If `truncated_fraction` is non-zero, the
comparison is invalid.

Then aggregate:

```bash
python -m evaluation.rlm.score evaluation/results/loft128k
python -m evaluation.compare --dataset loft --csv loft128k.csv
```

Sanity check: vanilla EM should land near the published Qwen3-8B LOFT-128k
baseline (`nq_128k` EM 0.40 with YaRN-4), allowing for 4B < 8B — though this model
has a real advantage there in not needing rope scaling at all.

## Known scoring quirks

Both are faithful to LOFT-official and deliberately left alone; they are recorded
here so the numbers are read correctly.

1. **`qampari` and `quest` reward a list repr over prose.** LOFT asks for the
   answer as a bracketed list. `extract_prediction` returns the whole cued tail as
   ONE answer, so `Final Answer: France, Spain, Italy` scores 0 coverage against
   three golds, while `['France', 'Spain', 'Italy']` scores 1.0. An arm that emits
   a list repr therefore has an edge on these two subsets that is about output
   format, not retrieval. Pinned by a test in
   `tests/evaluation/test_loft_answer_extraction.py`.
2. **A bracketed citation captures the answer slot.** `Final Answer: Paris [1]`
   extracts as `["1"]`. This is LOFT's own behaviour, since the bracketed list
   *is* the expected answer format.

What was *not* left alone: the RLM arm used to score 0.0 on every LOFT row
regardless of correctness, because its bare `str(FINAL(x))` matched neither
extraction path. `evaluation/benchmarks/loft/answer_extraction.py` now runs as a
fallback when the primary extractor returns nothing.

## LOFT coverage

RAG only — `nq`, `hotpotqa`, `musique`, `qampari`, `quest`. LOFT's text-retrieval
(13 datasets, recall@1) and many-shot ICL (5 datasets, em) categories have no
loader, no Hub mirror and no scorer here; they are a separate piece of work. See
`LOFT_TASKS` in `evaluation/benchmarks/registry.py` for what is actually valid as
`--data-dir`.
