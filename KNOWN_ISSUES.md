# Known issues

Things found during the 2026-08-29 refactor and deliberately **not** fixed, with
the reason. Everything here is a live limitation, not a TODO someone forgot —
if you hit one of these, this file is the explanation.

For what *was* fixed, see the `refactor/nim-removal-cleanup` commits. For the
earlier RLM audit, see `fix_rlm.md`.

## Test failures that are expected

`python -m pytest tests/` reports **42 failures** on a CPU-only machine. All of
them predate this refactor and none are caused by it (verified by diffing the
failure list against the pre-refactor baseline).

| Tests | Cause |
|---|---|
| `test_decoding_compression.py` (36) | `DecodingPress` / `CAMPress` return a compression ratio of exactly 1.0 — they do not compress at all. Real bug, but it lives in `decoding_press.py` / `cam_press.py`, which are byte-identical to upstream kvpress 0.5.4 and deliberately out of scope. |
| `test_pipeline.py` (5) | The tests assert log lines (`"Context Length: 23"`, `"Compressed Context Length: 13"`) that this fork's `pipeline.py` has never emitted — not even at the import commit. Upstream tests paired with a different upstream version. |
| `test_head_compression.py` (1) | `test_dms_press_compression_ratio`, same class of upstream issue. |

The refactor **fixed** 24 previously-failing tests (`test_presses.py` ×22,
`test_per_layer_compression_press.py`, `test_duo_attention_press.py`) by
propagating the model adapter through wrapper presses.

Also pre-existing: `evaluate.py`, `pipeline.py`, `sizing.py` and
`run_baseline.py` do not satisfy `black --line-length 120`. They were already
non-compliant before this work; only the hunks touched here were formatted, to
keep the diff reviewable. `make style` was already failing on master for
`evaluation/benchmarks/synthetickv*/generate_dataset.py`.

## Deliberate design limitations

- **KVzipPress masks, it does not free.** Evicted KV is hidden behind an
  attention patch, so a compressed run still needs the *full uncompressed* KV
  resident on the GPU. The memory budget controls how many tokens are logically
  retained — the same simulated-budget semantics as the LOFT/RULER matrix runs —
  not actual memory saved. This is why `--sub-min-free-gib` must cover the
  uncompressed context.
- **Memory units are decimal.** `MB` is 1000² and `GB` is 1000³
  (`kvpress/pipeline.py`). So `1024 MB` is **not** `1 GB` — it is 2.4% larger and
  lands in a different results directory. `SYNTHETIC_EXTENDED_BUDGETS` and
  `SYNTHETIC_ALL_BUDGETS` in `matrix_constants.py` look redundant for this
  reason but are genuinely different sweeps; merging them would reprice every run
  and orphan existing result directories.
- **Auto chunk sizing is not reproducible across hosts.** It reads live GPU free
  memory, so the same command on a busier GPU resolves to a smaller chunk. The
  resolved size is recorded in `config.yaml` as `max_subcall_chars_resolved`, and
  a resume that would drift by more than 2% aborts rather than silently merging
  two differently-sized experiments.
- **`_mentions` cannot ground a bare multiple-choice letter.** A model answering
  `B` on an `(A)/(B)` question is credited whenever `B` appears anywhere in the
  context. String matching cannot fix this class; it needs a task-aware scorer.
- **The results directory name is prefixed `new_`.** A legacy migration artifact
  in `EvaluationConfig.get_results_dir`. Removing it would orphan every existing
  result directory, so it stays.
- **`sub_model` is not in the RLM run-directory name.** Adding it would rename
  every directory ever written and orphan their checkpoints; it is enforced on
  resume via `RESUME_CRITICAL_KEYS` instead.

## Smaller things, knowingly left alone

- `rlm.py` observation truncation reports `len(out) - obs_limit` dropped, but
  builds `2 * (obs_limit // 2) + marker`, so for an odd `obs_limit` the reported
  figure is off by one.
- `kvzip_backend.py` truncates by decoding `ids[:max_context_tokens]` and then
  reuses the requested cap as the recorded `context_tokens`. A decode→re-encode
  round trip is not token-count preserving, so the recorded figure can drift
  slightly from what the model actually saw.
- `run_matrix.py`'s `rope_scaling` check requires an explicit `null` but its
  error message ("Profile requires rope_scaling: null") reads as though the key
  were missing.
- `memory_budget` means two different things across backends: the RLM root's
  eviction budget, and KVPress's KV compression budget. `compare.py` reads both
  under that name and `config.yaml` disambiguates with the `sub_kv_` prefix.
- Estimated vs. served token counts can disagree: the harness estimates locally
  with tiktoken while the server reports its own tokenizer's count. The
  server-reported number is preferred where available (`Usage.last_prompt_tokens`).

## LOFT campaign tooling (`evaluation/rlm/loft128k/`)

Found in the 2026-08-29 pre-campaign audit. The blocking and GPU-safety ones were
fixed (need-based `util_for` + `ROOT_NEED_MIB`, `MIN_FREE_MIB` 30000→33000,
`KVZIP_PYTHON` honouring an external `VENV`, `RLM_SUB_GPU` on the `run` path,
`download_data.sh` no longer forcing `$HOME`). These were left:

- **`flock` key omits `LENGTH` and the arm-4 budget/arm.** `run_cells.sh` locks on
  `.lock.$DATASET[.kvpress]`, so a 32k and a 1m run of the same subset collide
  unless they use different `RESULTS`, and the arm-4 grid can only be
  dataset-parallel, never budget-parallel. The lock also lives on NFS, where
  `flock` semantics are version-dependent — verify with a deliberate double-launch
  before trusting it as a duplicate-run guard.
- **Arms 1–3 always exit 0.** Both invocations end in `|| echo "WARN..."`, so
  `auto`'s `FAILED` accounting is blind to them; only the arm-4 lane propagates
  failure. A dead server shows up as a fast, silent, empty run.
- **`cleanup` never kills the worker subshells**, only the servers (`PIDS`, not
  `WPIDS`), and `pkill -TERM -P` is one level deep, so vLLM EngineCore
  grandchildren can survive an abort. Terminal Ctrl-C happens to work because
  SIGINT reaches the whole foreground process group; `kill <pid>` does not.
  Check `nvidia-smi` after any abort.
- **`kvzip-baseline` (cell 5) has no lock and no trap**, and inherits
  `memory_budget_unit`, `seed`, `model_kwargs` and
  `synthetic_kv_metadata_override` un-overridden from
  `evaluation/evaluate_config.yaml` — whose `model:` is a path on a host that is
  no longer reachable (masked only because the script overrides `model`).
- **README arm-4 smoke commands pass the retired `KV_RATIOS`**; the code reads
  `KV_BUDGETS`, so those "one-example smokes" silently run the default
  `0.512 1` grid — twice the intended work.
- **Arm 4c is missing from the README arm table** despite existing in
  `run_cells.sh` (`--max-subcall-chars auto` + `--target-compression-ratio`).
  It is documented only in `RLM.md`.
- **Stale comment in `run_infolab.sh`'s colocation branch** claims an overlong
  root transcript "400s, is recorded as an error, and is retried on resume". The
  code evicts and retries (`rlm.py`, counter `overflow_evictions`).
- **README's clone instruction names a branch that is not current.**

`slurm/loft128k_a100.slurm` is self-labelled UNVERIFIED and has never been
submitted; it also traps only `EXIT` (orphaning vLLM) and stages caches under
`$HOME`. Do not copy it as a template.

## Upstream kvpress bugs (unmodified files, not fixed here)

Found during the audit, left in place because these files are byte-identical to
upstream and keeping them so makes rebasing on a future kvpress release possible:

- `base_press.py` calls `logger.warning_once(...)` on a stdlib logger, which has
  no such method — any `Gemma3ForConditionalGeneration` run raises
  `AttributeError` before a hook is registered. (This one *is* in a modified
  file; it is upstream code that no current model path reaches.)
- `FastKVzipPress` sizes `score_val` to every layer but skips sliding-window
  layers, leaving `None` entries that `torch.stack` then rejects on Gemma3. It
  also bypasses the Qwen3.5 support check entirely, since it defines its own
  `__call__`.
- `search_hyperplane`'s rescale only ever shrinks the vector, so a
  small-magnitude input can yield a hyperplane too weak to drive
  `exp(<q,k>)` to zero — masking then fails silently instead of raising.
- Five presses reach into `cache.layers[...]` directly instead of going through
  the model adapter, so the adapter abstraction is only half-applied.
- `kvzip_press.py` restores its reconstruction state at the *top* of each chunk
  loop rather than after, which is correct only by accident of ordering, and
  deep-clones the whole KV cache per chunk.

## Things the plan proposed that were rejected on inspection

- **Consolidating the synthetic budget tuples** (see the decimal-units note
  above) — they are not duplicates.
- **Reusing one `ModelAdapter` instance across `pipeline.py`** — the adapter is
  a stateless object whose construction is a string comparison plus an empty
  `__init__`; threading it through `generate_answer` would change a public
  signature for no measurable gain.
