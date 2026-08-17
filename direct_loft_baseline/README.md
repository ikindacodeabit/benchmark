# Direct LOFT Baseline Validation (legacy)

The generalized replacement is [`../direct_baseline/README.md`](../direct_baseline/README.md).
Use that runner for new work; it supports every registered dataset and generic
Hugging Face models while retaining the same no-KVPress inference boundary.

This directory provides an independent LOFT baseline for checking the KVPress
`no_press` results. It intentionally does not import the `kvpress` package,
create a press, register hooks, construct a KVPress cache, or apply a memory
budget. Every question is evaluated by a normal Hugging Face `generate()` call
over the complete context and question.

The prompt is tokenized in the same two parts used by the benchmark pipeline:
the chat-templated context, followed by the question, assistant suffix, and
dataset answer prefix. Qwen3.5 is loaded text-only through
`Qwen3_5ForConditionalGeneration`. The model creates and manages its native
cache internally.

## Direct baseline run

Run this only on a GPU compute node:

```bash
python run_direct_loft.py \
  --model /home/rethinkingai-self/25m0820/kvpress/Qwen3.5-27B \
  --tasks nq_32k hotpotqa_32k musique_32k qampari_32k quest_32k \
  --output-dir outputs/qwen35_27b_loft32k \
  --device cuda:0
```

For a short smoke test, add `--limit 3`. Thinking is disabled through the
official chat-template argument by default. Use `--enable-thinking` only when
you intentionally want thinking output.

Each task produces:

- `predictions.jsonl`: resumable per-question output
- `predictions.csv`: comparison-friendly output
- `metrics.json`: LOFT EM, subspan EM, and F1/coverage
- `run_summary.json`: prompt and think-tag statistics

## Inspect thinking blocks

```bash
python scan_think_blocks.py \
  /path/to/results_a /path/to/results_b \
  --json-output think_report.json
```

The scanner reads only the `predicted_answer` column and reports rows with
`<think>` or `</think>`, including unclosed blocks.

## Compare against a KVPress no-press result

```bash
python compare_baselines.py \
  --direct-csv outputs/qwen35_27b_loft32k/nq_32k/predictions.csv \
  --reference-csv /path/to/kvpress/no_press/predictions.csv
```

This comparison does not run KVPress. It only joins two saved prediction files
by task, split, and question and reports exact-output agreement and think-tag
rates.
