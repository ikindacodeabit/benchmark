# LOFT-128k: vanilla vs RLM vs RLM+scratchpad

Three arms on LOFT's five RAG subsets at 128k context, served by
`Qwen/Qwen3-4B-Instruct-2507`.

| arm | invocation | what it tests |
|---|---|---|
| `vanilla` | `--mode both` | the whole 131k-token document in the context window |
| `rlm` | `--mode both` | document held in a REPL variable, root model chunks and recurses |
| `rlm+scratchpad` | `--mode rlm --scratchpad` | same, plus a persistent `note()` buffer that survives eviction |

`--mode both` produces the first two arms in one pass, so the vanilla arm is not
recomputed for the scratchpad comparison. The three land in separate run
directories (the run-dir name carries a `scratchpad` component), and each is
resumable through its own `checkpoint.jsonl`.

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
