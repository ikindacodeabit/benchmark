# RLM Sub-Calls over a KV Memory Budget

This repository hosts two ways of surviving a context that does not fit in the model's
window, and one experiment that combines them. **KVPress** compresses the KV cache so the
whole document can be attended to at reduced cost. An **RLM** (Recursive Language Model)
never puts the document in the window at all: a root model drives a Python REPL in which the
document is a variable, and reads it in slices through a recursive `llm_query` sub-call. The
two keep entirely independent inference paths and touch at exactly one seam — the sub-call.

That seam is where the interesting arm lives. If sub-calls read their slice through a
compressed KV cache, compression stops being an alternative to recursion and becomes the
thing that makes recursion's reads cheaper. This document explains how the two systems
coexist here, why their scores are comparable, and how the size of a sub-call slice is
derived from the KV memory budget rather than picked by hand.

## Before and after

| Before | After |
|---|---|
| Sub-call chunk fixed at 32,000 chars, whatever the budget, model or GPU | Derived from the KV budget and a target compression ratio |
| The size was a magic number repeated in three files | One named constant, one pure function |
| Chunk size and `--memory-budget` set independently, free to contradict | Chunk follows from the budget; the contradiction cannot arise |
| Chars-per-token assumed to be 4.0 | Calibrated per run from the actual document |
| GPU fit factors lived only inside `_fits_in_memory` | Shared with the planner, so they cannot drift |
| Nothing recorded why a size was chosen | `subcall_sizing` in `config.yaml` names every cap considered |
| Sub-context truncation was silent | The root is told, as it already is for the char cap |

---

## 1. How RLM and KVPress coexist

Files: [`evaluation/rlm/rlm.py`](evaluation/rlm/rlm.py),
[`evaluation/rlm/kvzip_backend.py`](evaluation/rlm/kvzip_backend.py)

The harnesses share everything except inference:

| Concern | Shared implementation |
|---|---|
| Which datasets exist | `evaluation/benchmarks/registry.py` |
| Loading and schema normalisation | `evaluation/benchmarks/loaders.py` |
| Scoring | `benchmarks/results.score_prediction_frame` |
| Result contract | `predictions.csv`, `metrics.json`, `config.yaml` |
| Cross-backend comparison | `evaluation/compare.py` |
| Inference | **independent** — vLLM/NIM over HTTP for RLM, local HF pipeline for KVPress |

The seam is two methods. `rlm.py` consumes a client that offers `.chat(messages)`; a
KV-compressing client additionally offers `.chat_split(question, context)`. When the sub
client has the second method, the REPL tool the root is taught changes shape:

```python
# dense (nim backend): one argument, the slice pasted into the prompt
llm_query("Answer X based on this text:\n" + context[i:j])

# split (kvzip backend): two arguments, the slice compressed separately
llm_query("What is X?", context[i:j])
```

`RLM.run` sets `split_capable = hasattr(self.sub, "chat_split")` (`rlm.py:556`) and selects
the matching help text and worked example into the system prompt (`rlm.py:565-580`);
`llm_query` itself folds the two arguments back into one prompt when the client cannot take
them apart (`rlm.py:323`). The dense help text is **byte-pinned** by
`tests/evaluation/test_rlm_subcall_context.py::test_dense_prompt_rendering_is_pinned`,
because arms 1–3 of the LOFT campaign are in flight and a wording change mid-campaign would
confound them. The split help text is a format string and is free to change.

### The compression here is logical, not physical

This is the single most misreadable thing in the arm, and it is stated in the backend's own
module docstring. `KVzipPress` masks evicted keys via an attention patch; it does **not**
free them. A slice therefore needs its **full uncompressed KV resident for the whole call**,
whatever the budget says.

So: the memory budget buys a *retention target*, not headroom. A bigger chunk costs full GPU
memory. Sizing chunks from the budget makes the realized compression ratio constant and
comparable across budgets — it does not let a sub-call afford a bigger read for less memory.
Anything that reads like "compression enables bigger reads" is a claim about the *method*,
not about this implementation's memory use.

---

## 2. Why the two backends' numbers are comparable

File: [`evaluation/compare.py`](evaluation/compare.py)

Two properties make a KVPress row and an RLM row commensurable:

- **The same scorer.** Both call `score_prediction_frame` on the same gold answers, so
  `score` is literally the same function applied to the same data.
- **The same cost axis.** `compare.py` maps both onto `context_tokens_held`: KVPress's
  `average_retained_context_tokens` and RLM's `average_peak_context_tokens`. Both answer
  "how much context was live at once", which is what KV memory is proportional to.

Two asymmetries are recorded rather than hidden, because ignoring either flatters RLM:

- **Errors.** An RLM example that dies on an API error is dropped from the score and counted
  in `runtime.errors`. KVPress runs locally and cannot fail this way, so scoring an outage as
  zero would penalise RLM for something unrelated to the method. Check `errors` before
  believing a score.
- **Truncation.** KVPress compresses the whole context; vanilla truncates it.
  `context_retained` is the surviving fraction, so a low vanilla score can be read as a
  property of the char limit rather than of the model.

---

## 3. The memory-budget math that already existed

File: [`kvpress/pipeline.py`](kvpress/pipeline.py)

Budgets are **decimal**, and must stay that way to match the published LOFT/RULER numbers in
`evaluation/matrix_constants.py`:

```python
MEMORY_UNIT_TO_BYTES = {"MB": 1000**2, "GB": 1000**3}
```

Per-token KV cost comes from the model adapter — layers × 2 (K and V) × KV heads × head_dim ×
bytes per element. For `Qwen/Qwen3-4B-Instruct-2507` in bf16 that is
36 × 2 × 8 × 128 × 2 = **147,456 bytes/token** (144 KiB), so a 1 GB budget retains
1,000,000,000 ÷ 147,456 = **6,781 tokens**.

The budget then becomes a ratio, per call:

```python
retained_context_tokens = min(context_length, token_budget)
compression_ratio = 1 - (retained_context_tokens / context_length)
```

Two consequences worth internalising:

1. `context_length` is **this call's actual slice**, and `_forward` *mutates*
   `press.compression_ratio` on every call. One press instance is re-tuned per example.
2. If the budget already covers the slice, the ratio is exactly `0.0` — no error, no skip.
   The press runs and does nothing.

---

## 4. Deriving the sub-call chunk size

File: [`evaluation/rlm/sizing.py`](evaluation/rlm/sizing.py)

Point (1) above is the lever. The realized ratio depends on how big a slice the root sends,
so if the slice size is chosen by hand and the budget by sweep, the two drift and every arm
measures a different effective compression. Reading the formula backwards fixes the ratio
instead:

```
chunk_tokens = token_budget / (1 - target_compression_ratio)
chunk_tokens = min(chunk_tokens, sub_window - reserve, cli_cap - reserve, gpu_fit)
chunk_chars  = chunk_tokens * calibrated_chars_per_token
```

The caps, in the order they usually bind:

| Cap | Where it comes from | Why |
|---|---|---|
| `budget` | `token_budget / (1 - target)` | the ask |
| `cli_cap` | `--sub-max-context-tokens` − reserve | the operator's explicit ceiling |
| `sub_window` | the model's `max_position_embeddings` − reserve | hard architectural limit |
| `gpu_fit` | `(free_bytes − 1 GiB) / (1.2 × kv_bytes_per_token)` | full uncompressed KV must fit |

`reserve_tokens` (default 1024) exists because the sub client truncates only the *context* to
its token cap and then appends the question and decodes on top of it — nothing else budgets
for those two. `gpu_fit` is the exact inverse of the per-call check in
`kvzip_backend._fits_in_memory`, sharing `gpu_fit_token_cap` so that the size advertised to
the root and the size accepted at call time cannot drift apart. Note it uses **binary** GiB:
it measures device memory, not a benchmark axis.

### Worked example — Qwen3-4B, 1 GB budget

`token_budget` = 6,781 tokens. Calibrated at 3.6 chars/token:

| target | `--sub-max-context-tokens` | wanted | chosen | binding | ratio if filled | chars |
|---|---|---|---|---|---|---|
| 0.5 | 131,072 | 13,562 | 13,562 | `budget` | 0.500 | 48,823 |
| 0.75 | 131,072 | 27,124 | 27,124 | `budget` | 0.750 | 97,646 |
| 0.9 | 131,072 | 67,810 | 67,810 | `budget` | 0.900 | 244,116 |
| 0.9 | 34,000 | 67,810 | **32,976** | `cli_cap` | **0.794** | 118,713 |

The last row is the point of recording `binding`: the run asked for 0.9 and can only reach
0.794, and the size alone does not say so.

### It reproduces the arms that were picked by hand

The two existing arm-4 lanes were bracketed by intuition. Run the derivation against them:

| hand-picked | closest derived setting | derived chars |
|---|---|---|
| 4a — 32,000 chars | 0.512 GB @ target 0.5 | 27,776 |
| 4b — 131,072 chars | 1 GB @ target 0.75 | 108,496 |

Both land within ~20% of the hand-picked value, which is the evidence that the formula is
measuring the same thing the original arms were reaching for — and it names the budget each
was implicitly encoding.

### Chars, not tokens

The root slices with `context[i:j]`, in characters, and the harness truncates in characters —
so the cap advertised to the model has to be a character count, while the budget is in
tokens. The conversion is measured, not assumed: `calibrate_chars_per_token` tokenizes a
sample from the **middle** of a real document (both LOFT and RULER open with boilerplate that
tokenizes unrepresentatively), clamps the result to `[1.5, 6.0]`, and records which of
`calibrated` / `clamped` / `fallback` happened. LOFT prose runs ≈3.6–4.2; dense subsets like
RULER `cwe` run nearer 2.5, and the old fixed 4.0 overshoots them by ~60% — which used to
mean the sub client silently dropped the tail of every slice.

---

## 5. Using it

```bash
python -m evaluation.rlm.run_benchmark \
    --dataset loft --data-dir nq_128k --mode rlm --scratchpad \
    --sub-backend kvzip \
    --memory-budget 1 --memory-budget-unit GB \
    --max-subcall-chars auto --target-compression-ratio 0.75
```

Through the LOFT runner this is arm **4c**, alongside the hand-sized 4a and 4b:

```bash
KVPRESS_ARMS=1 KV_ARMS=4c KV_BUDGETS="0.512 1" KV_TARGET_RATIO=0.75 \
    DATASETS=nq LENGTH=32k LIMIT=1 SERVERS=1 \
    bash evaluation/rlm/loft128k/run_infolab.sh auto
```

`KV_BUDGETS` replaces the old `KV_RATIOS` for the arm-4 lanes — the press takes a budget and
derives each call's ratio itself. (`KV_RATIOS` still drives the *cell-5* `evaluate.py`
baseline, which genuinely does take `--compression_ratio`.) 4c is a third arm rather than a
retargeting of 4a/4b, whose results are mid-campaign.

`auto` requires `--sub-backend kvzip` and a target ratio; the `nim` path has no KV budget and
no local tokenizer, so a derived number there would be fabricated. The ratio must be in
`[0.0, 1.0)` — 1.0 asks for an infinite chunk.

**`auto` is not the default**, deliberately. Turning it on would change the number rendered
into the split prompt for every in-flight arm, change every run-directory name, and make the
size depend on a live GPU reading. The default stays the literal 32,000.

The size is resolved **once**, before the RLM is built, from one document pulled out of the
dataset — then frozen. It is rendered into the root's system prompt, so re-deriving it per
example would vary the prompt within a single run.

### Run directories and resume

`autosub<ratio>` is appended after the existing `sub<N>` component, so every hand-sized run
directory keeps exactly the name it had before this feature existed:

```
...__rlm__scratchpad__kvzip-kvzip1GB                        # fixed, default size
...__rlm__scratchpad__kvzip-kvzip1GB__sub131072             # fixed, hand-raised
...__rlm__scratchpad__kvzip-kvzip1GB__sub108496__autosub0.75  # auto
...__rlm__scratchpad__kvzip-kvzip1GB__autosub0.9            # auto, resolved to 32000
```

That last line is why the marker exists: an auto run that happens to resolve to exactly the
default size carries no `sub<N>` component, so the marker is the only thing stopping it from
resuming into the hand-sized checkpoint.

The name carries the *ratio*, not the resolved size — otherwise GPU-occupancy jitter would
fragment every resume into a fresh directory and auto runs would never resume at all. The
residual risk (a resumed auto run resolving differently and merging two experiments) is
caught explicitly: on resume the recorded `max_subcall_chars_resolved` is compared against
the fresh resolution and the run **stops** if they differ by more than 2%.

---

## 6. What gets recorded

`config.yaml`:

| Key | Meaning |
|---|---|
| `subcall_sizing_mode` | `fixed` or `auto` |
| `target_compression_ratio` | what was asked for (`null` when fixed) |
| `max_subcall_chars` / `max_subcall_chars_resolved` | the size actually used |
| `subcall_sizing` | the full plan: `token_budget`, `binding`, every entry of `caps`, `chars_per_token` and its source, `realized_ratio_if_filled` |

`metrics.json → runtime`:

| Key | Target or realized |
|---|---|
| `subcall_chars_advertised`, `subcall_target_compression_ratio`, `subcall_sizing_binding` | target |
| `average_sub_context_tokens` | realized — how big the root's slices actually were |
| `average_sub_compression_ratio` | realized — what the press actually did |
| `sub_split_call_fraction` | did the root use the two-arg form at all |
| `sub_pressed_call_fraction` | did the press actually engage |

Read them in pairs. `sub_pressed_call_fraction` near 0 means the arm degenerated to dense
calls and measured nothing. `average_sub_compression_ratio` well below
`subcall_target_compression_ratio` means the root under-filled the chunk.

---

## 7. Known issues

Recorded so they are not rediscovered as bugs. The list was longer; see
[`fix_rlm.md`](fix_rlm.md) for what was fixed and why.

1. **`KVzipPress` masks rather than frees.** A bigger chunk costs full GPU memory. Sizing
   from the budget buys comparability, not headroom. This is why the reported cost axis is
   `max(root_peak, sub_peak)` — the sub model's KV is resident whatever the budget says.
2. **The advertised cap is a suggestion.** `rlm.py` truncates slices on the way up but never
   pads them, and the press derives its ratio from what actually arrived. A root that sends
   small slices produces a realized ratio near 0 no matter what was advertised.
3. **Decimal GB vs binary GiB coexist on purpose.** Budgets are decimal (they must match the
   published matrix numbers); the GPU fit headroom is binary (it measures device memory).
4. **Auto sizing is not bit-reproducible across machines.** `mem_get_info` is a device-global
   reading that a co-tenant process changes — hence the recorded resolved value and the
   resume check.
5. **Venv conflict.** The backend imports `kvpress`, which wants `transformers>=4.56`, but the
   main `.venv` is pinned to `4.51.3` because vLLM 0.8.5 calls `all_special_tokens_extended`,
   removed in 5.x. Two venvs remain the answer until the vLLM pin can move.
6. **The root's context estimate is not the server's.** `TokenCounter` is tiktoken or
   `len // CHARS_PER_TOKEN_ESTIMATE`, and `count_messages` approximates chat overhead at
   4 tokens per message. `peak_context_tokens` prefers the server's own count when the
   client reports one, but the eviction decisions that *precede* a call are still made on
   the estimate — which is why the server gets the final say via the overflow retry.

---

## See also

- [`evaluation/rlm/README.md`](evaluation/rlm/README.md) — the RLM harness itself: guards,
  scratchpad, benchmarks, metrics.
- [`evaluation/rlm/loft128k/README.md`](evaluation/rlm/loft128k/README.md) — the LOFT-128k
  arm-by-arm runbook.
- [`fix_rlm.md`](fix_rlm.md) — the audit that produced the generation-2 changes: what was
  broken, what was fixed, and what is deliberately left alone.
- [`evaluation/README.md`](evaluation/README.md) — the KVPress evaluation matrix.
