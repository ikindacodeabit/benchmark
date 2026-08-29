# fix_rlm.md — catalogue of RLM implementation fixes

> **Status: applied.** Every item below except the "keep documented" list was
> implemented on 2026-08-27, in one pass rather than the staged batches suggested at
> the end. 115 tests pass (`pytest tests/evaluation/`), mypy and flake8 are clean on
> the changed files, and an end-to-end stub run confirms the four behaviours unit
> tests do not wire together: the answer-format contract reaching both arms, the
> resume guard refusing a changed `--limit`, checkpoint de-duplication after a
> retried error, and abstention landing in `runtime.abstained`.
> The catalogue is kept as the record of WHY each change was made.

Audited 2026-08-27 on branch `rlm-budget-derived-chunk-size` (HEAD `16e191d`), against
the standalone repo `rlm-prajna2-git` (branch `fix/rlm-bugs-prajna-readiness`, which is
ahead on RLM-loop bug fixes) and the completed LOFT-128k campaign results.

**Comparability decision:** the LOFT-128k arms 1–3 campaign (completed 2026-08-27) is
closed and archived. Everything below — including prompt-wording and grounding-semantics
changes that were deliberately frozen "mid-campaign" — is now fair game. Post-fix runs
are **generation 2** and must not be compared row-for-row with generation-1 results.

**Relationship to `RLM.md` §7:** that section's 12 known-issues items remain accurate.
Items absorbed here are cross-referenced as *(RLM.md §7.n)*; the three that are
documentation-of-design rather than defects are listed at the end under "Keep documented,
don't fix".

**Priorities:** P0 = silently corrupts results or kills runs; P1 = costs score or
measurement fidelity; P2 = hygiene/robustness.

---

## A. Ports from `rlm-prajna2-git` (fixes already written there — mine, don't reimplement)

The vendored copy at `evaluation/rlm/rlm.py` forked before these landed in the
standalone repo's `rlm/rlm.py`. Each is a `git show` away.

### A1. Grounding check is a raw substring test — P1
`evaluation/rlm/rlm.py:674` (`val in seen_output or val in task`) and `:763`.
Any short literal contained *anywhere* in the task string is accepted as grounded:
a question containing "12" licenses an invented `FINAL("12")`, and every
classification / multiple-choice task enumerates its labels in the prompt, so that
entire task class has no grounding at all.
**Fix:** port `_mentions()` (word-boundary regex containment) from the standalone repo
and use it for the `task` side of both checks. Standalone's docstring explains the
failure class.

### A2. No recovery for a code fence truncated at max_tokens — P1
When the root's reply is cut mid-code-block, `CODE_RE` (`rlm.py:160`) finds nothing,
the turn is treated as "no code block", burns a nudge (`rlm.py:719`), and three such
replies return the raw truncated text as the answer (`gave_up_no_code`, `rlm.py:728-736`).
**Fix:** port `UNTERMINATED_CODE_RE` (an opening fence with no closing one) plus the
`metrics["truncated_code_blocks"]` counter from the standalone repo.

### A3. `FINAL_VAR` on a missing variable is recorded as a successful answer — P1
- Prose path: `rlm.py:709` — `env.get(name, f"<missing var {name}>")`, returned
  immediately as `final_var_in_prose` with `finished=True`.
- In-code path: `FINAL_VAR` in `_make_env` has the same `.get` default.
The literal string `<missing var x>` becomes the scored prediction.
**Fix:** port the standalone behavior — raise `NameError` into the REPL observation
(in-code) / reject-and-nudge (prose) so the model sees the mistake and can correct it.

### A4. No `answer_format` parity between arms — P1
Standalone threads the benchmark's output contract (LOFT's `Final Answer: [...]`,
carried per-example as `answer_prefix`) into **both** arms: `RLM.run(answer_format=)`
renders it into `ROOT_SYSTEM_PROMPT` and `vanilla_answer(answer_format=)` appends
"Format your answer as: ...". The vendored copy has neither, so vanilla and RLM are
asked for different answer shapes and string metrics measure style.
The vendored repo partially compensates downstream with
`evaluation/benchmarks/loft/answer_extraction.py` (RLM-aware extraction fallback,
`calculate_metrics.py:245-256`).
**Fix:** port the `answer_format` plumbing *and keep* the extractor — the prompt-side
contract raises the hit rate of the cheap cued path; the extractor stays as the safety
net. Also port standalone `0920a87` ("do not stack a format instruction the question
already carries") so the contract isn't injected twice.

### A5. `note` missing from `STMT_KEYWORDS` — P2
`rlm.py:161`. `ONELINE_FIX_RE` repairs one-line replies like `x = 1 print(x)`, but
scratchpad `note(...)` calls jammed onto one line aren't split because `note` isn't in
the keyword alternation. Standalone has it.
**Fix:** add `note` to the alternation (harmless when scratchpad is off).

### A6. Scoring/reporting parity with standalone — P1 (verify, then port the gaps)
- `22e3112` "errored record is a missing measurement, not a zero": the core exclusion
  **is already present** (`run_benchmark.py:104-111`) — verify the abstention and
  tok/q / s/q denominators match too (standalone also excludes errored rows from those).
- `c94a7cb` "report the vanilla arm's truncation ceiling beside its score": vendored has
  `truncated_fraction` / `average_context_chars_retained` (`run_benchmark.py:147-153`) —
  close, but confirm the per-subset ceiling column exists in the compare output, not
  just the run metrics.
- Standalone's vanilla-arm commits `1c09cb3` (server decides whether the prompt fits),
  `3f1cf2b` (token ceiling, not just char limit), `f71c70b` (char limit must not cripple
  the baseline): the vendored `vanilla_answer` (`rlm.py:885-954`) has a version of the
  shrink-retry loop; diff and reconcile so the two repos' vanilla arms are the same arm.

---

## B. Design adaptations vs the original RLM paper (what the campaign data argues for)

### B1. Abstention channel — P0 for the next campaign
The grounding guard rejects legitimate abstentions. `FINAL("unknown")` /
`FINAL("No relevant information found")` is by construction absent from REPL output
(nothing was found — that's the point), so it hits the literal check at `rlm.py:763`,
gets rejected up to three times, and the example ends `ungrounded_final` with
`pred=None`, scored 0 via the `""`-fill (`run_benchmark.py:104-109`).
In the LOFT-128k campaign this was 7–39% of examples per subset (musique worst, 42/110)
— those answers score 0 either way *today*, but the transcript misreports a correct
abstention as a hallucination-guard rejection, and any future abstention-aware metric
is impossible.
The design is also self-contradictory: the system prompt *forbids* concluding
"the answer is 0 or empty" (`rlm.py:61-64`), yet musique contains bridges genuinely
unresolvable from the document.
**Fix:**
1. Add an explicit abstention primitive — `FINAL_NONE(reason)` (or whitelist a small
   set of abstention phrases in the guard) — terminating cleanly with
   `end_reason="abstained"`, distinct from `ungrounded_final`.
2. Teach it in `ROOT_SYSTEM_PROMPT` and soften `rlm.py:61-64` to allow abstaining
   *after* a genuine search effort.
3. Close the guard's inconsistencies while touching it:
   - only AST string literals are checked (`rlm.py:450-457` collector): `FINAL(var)`,
     `FINAL(f"...")`, `FINAL("a"+"b")` all bypass; a model that writes
     `ans="unknown"; FINAL(ans)` is accepted while `FINAL("unknown")` is not;
   - the prose `FINAL_VAR` path (`rlm.py:707-718`) has **no** grounding check at all;
   - truncated observations poison the accumulator: `seen_output` stores the
     head+tail-truncated obs (`rlm.py:501-504` → `:759`), so an answer that appeared
     only in the elided middle reads as ungrounded. Ground against the *untruncated*
     output (keep a separate accumulator, capped much higher).

### B2. Root context is unbounded in arms 1–3 — P0
`loft128k/run_cells.sh` never passes `--max-context-tokens` for arms 1–3, so
`budget is None` and `build_sent` (`rlm.py:592-647`) sends the full history, growing
monotonically for up to `--max-steps 50` turns. `NIMClient` does not retry 4xx
(`client.py:112-120`), so a root-context overflow becomes `end_reason="exception"`
(`run_benchmark.py:624-627`). This is the same mechanism that zeroed the RULER-32k
Qwen3-8B RLM run (every sample `end_reason=exception`). Meanwhile the vanilla arm has
an elaborate server-authoritative shrink-and-retry loop (`rlm.py:917-954`,
`_is_context_overflow` `rlm.py:872-882`) — the root path has no equivalent.
**Fix:** either (a) default a root `MemoryBudget` sized from the served window, or
(b) catch `_is_context_overflow` on `self.root.chat(sent)` and retry after evicting
oldest turns (reusing the existing squeeze loop). (b) is more faithful to "unbounded
until the server objects" and needs no new tuning knob.

### B3. Cross-backend cost axis omits sub-call KV — P1
`peak_context_tokens` measures the **root** prompt only (`rlm.py:656-658`), and
`compare.py:132` maps it to `context_tokens_held` against KVPress's
`average_retained_context_tokens`. Arm 4's sub model concurrently holds up to ~33k
tokens of (logically-compressed but physically resident) KV, reported separately as
`runtime.average_sub_context_tokens` and never folded in — the headline "context held"
for the kvzip arm is flattering.
**Fix:** report `max(root_peak, sub_peak)` (or both columns side-by-side) in
`compare.py`; at minimum footnote the asymmetry in the table.

### B4. Token accounting is estimated where it could be measured — P2
`root_prompt_tokens` / `peak_context_tokens` come from `TokenCounter` (tiktoken or the
`len//4` fallback, `rlm.py:192`), while the server's true usage is available in
`client.usage` (`client.py:24-29`) and used only for the `tokens` column
(`run_benchmark.py:630`). A second uncalibrated constant lives at `rlm.py:667`
(`budget.max_context_tokens * 3` chars — mixing a token budget with a char limit).
*(RLM.md §7.12.)*
**Fix:** prefer server-reported prompt tokens for the peak metric when available;
calibrate the two constants with `sizing.calibrate_chars_per_token`, which already
exists for the sub-call path.

### B5. Split sub-call can carry ~2× the advertised cap — P2 *(RLM.md §7.3)*
Question and context are capped **independently** at `max_subcall_chars`
(`rlm.py:340-346` area; cap logic at the `truncated = ...` block), and the truncation
notice is appended *after* the question cap is applied. A capped-out split call can
therefore exceed the sub GPU fit check, whose failure returns a
`[SUB-MODEL ERROR] ... retry with a smaller snippet` *string* the model may ignore.
**Fix:** cap the (question + context) *sum*; append the notice before final capping;
make repeated fit-check failures count toward the nudge/abort machinery instead of
looping forever on model goodwill.

### B6. kvzip `chat()` compresses its own system prompt — P1 for arm-4 fidelity
`kvzip_backend.py:250-259`: the one-arg path flattens **all** messages —
`SUB_SYSTEM_PROMPT` included — into the compressible `context` with an empty question.
The repetition-breaker fallback and any dense-form `llm_query` against the kvzip
backend therefore expose the instruction text itself to KV masking.
**Fix:** route the system prompt through the uncompressed question side (as
`chat_split` at `:261-264` already does) and flatten only user content into context.

### B7. Unfreeze the byte-pinned prompt mismatches — P2 *(RLM.md §7.11)*
`LLM_QUERY_HELP_DENSE` says "keep each call under ~8000 characters" (`rlm.py:102`)
while enforcement is `max_subcall_chars = 32000` (`rlm.py` ctor default /
`run_benchmark.py:33`). The pin existed for arms 1–3 comparability; the campaign is
closed. **Fix:** render the real cap into the dense help (as the split help already
does via `{max_chars}`), and update the byte-pinning test
`tests/evaluation/test_rlm_subcall_context.py` to pin the *new* generation-2 text.

### B8. Small-scaffold cleanups — P2
- `repetition_broken` fallback (`rlm.py:798-839`) bypasses grounding by design — fine,
  but stamp its transcript entry so triage can tell forced answers from chosen ones.
- `_notes_block` (`rlm.py:529-535`) prepends a fresh `[oldest notes truncated]` marker
  every shrink iteration and cannot shrink below 200 chars regardless of
  `max_notes_tokens`; dedupe the marker and honor the cap.
- SIGALRM re-arm at `max(remaining, 0.05)` after a sub-call (`llm_query` timer
  save/restore) can abort a cell 50 ms after `llm_query` returns when the timer expired
  mid-call — consider granting a small grace re-arm (e.g. min 1 s) so sub-call latency
  isn't billed to the code watchdog.

---

## C. Harness / reporting fixes

### C1. Errored rows double-count on resume — P0
`load_done` (`run_benchmark.py:73-85`) excludes errored ids so they're retried — good —
but the stale error line stays in `checkpoint.jsonl` and the retry **appends** a second
record with the same id (`:633`). `write_run_artifacts` builds its frame from every
line with no de-dup (`:100`), so `runtime.examples` and `runtime.errors` are inflated
and `predictions.csv` carries duplicate ids. `errors` is exactly the column RLM.md tells
you to check before believing a score.
**Fix:** de-duplicate by id keeping the last record, in `write_run_artifacts` (and in
`load_done`'s accounting). One-line pandas `drop_duplicates("id", keep="last")` plus a
test in `test_run_benchmark.py::ErroredRecordTest`.

### C2. `--limit` unguarded on resume + the dev-split trap — P0
- `--limit` is re-applied on every run (`run_benchmark.py:552`) and is neither in the
  run-dir slug nor validated against the stored `config.yaml` (`limit` is written at
  `:657` but nothing reads it back — unlike the 2% `autosub` resume guard at
  `:534-547`). Resuming with a different limit silently merges two example sets into
  one checkpoint — the exact hazard `build_run_dir_components`'s docstring exists to
  prevent.
- `_load_loft` concatenates **dev (10) then test (100)** (`loaders.py:37-46`) and
  stamps a `split` column nothing reads, so `LIMIT ≤ 10` evaluates the dev split
  exclusively — smoke runs land in the results tree looking like real runs scoring
  0.000 (bitten in practice).
**Fix:** (a) hard-error on resume when `--limit` differs from `config.yaml`, same
pattern as the autosub guard; (b) add a `--split dev|test|all` filter to
`iter_benchmark_examples` and default real runs to `test`, smoke runs to `dev` — which
also makes `LIMIT=10` smoke runs land in a *differently-slugged* directory if split is
added to the slug.

### C3. Run-dir slug omits result-affecting knobs — P1
`build_run_dir_components` (`run_benchmark.py:205-236`) covers
dataset/task/model/mode/ctx/scratchpad/kvzip-budget/subN/autosub but not `--max-steps`,
`--run-timeout`, `--max-sub-calls`, `--sub-model`, `--press-min-tokens`,
`--sub-max-context-tokens`, `--vanilla-char-limit`. Lanes 4a and 4b differ in
`--max-sub-calls` and `--run-timeout` and are only separated because their
`--max-subcall-chars` happen to differ.
**Fix:** don't bloat the slug — extend the resume guard instead: on resume, compare the
full stored config against the live one and hard-error on any mismatch in a declared
list of result-affecting keys (slug stays stable, silent merges become impossible).
Add `sub_model` to the slug though — a different sub model is a different experiment.

### C4. `compare.py` cannot distinguish arms 2 and 3 — P1
`describe_configuration` (`compare.py:86-105`) never reads `config["scratchpad"]`
(faithfully written by `run_benchmark.py:662`), so plain-RLM and RLM+scratchpad rows
both render as `rlm`; the groupby sanity checks then treat them as duplicates of one
configuration. **Fix:** append `+scratchpad` (and the notes budget when non-default) to
the description.

### C5. Three conflicting default results paths — P1
`run_benchmark.py:417` writes `evaluation/results/rlm`; `compare.py:152` reads
`evaluation/results`; `score.py:15,80` reads `benchmark_artifacts/results/rlm` — and
`evaluation/rlm/README.md:54` + `QWEN3_LOCAL_RUNBOOK.md` document the latter. An
out-of-the-box `score.py` run prints an empty table.
**Fix:** one constant, imported by all three; update both docs.

### C6. Vanilla truncation stats lost on the error path — P2
`vanilla_answer` fills `stats` before re-raising (`rlm.py:948-950`) but the caller only
merges it on success (`run_benchmark.py:591`), so the errored record carries no
truncation exposure — precisely the record where you want it for triage.
**Fix:** merge `vstats` in the exception handler too.

### C7. `SCORE_KEYS` ordering + `progress_match` foot-gun — P2
`compare.py:36`: `subspan_em` sits behind `accuracy`/`score` in the priority list, so a
future scorer emitting both picks the wrong headline silently (the qampari/quest
dropout was already one instance of this class). And `runtime.progress_match`
(`run_benchmark.py:62-70`, loose substring) ships in the same `metrics.json` as the
canonical score; it is documented as non-authoritative but nothing stops a reader
grabbing it. **Fix:** per-dataset explicit headline key (LOFT → `subspan_em`); rename
`progress_match` → `progress_match_loose`.

### C8. Auto-sizing: `cli_cap` silently binds at high target ratios — P2
With the default `--sub-max-context-tokens 34000`, `plan_subcall_chunk` caps at
32,976 tokens, so for `KV_TARGET_RATIO ≳ 0.79` (at 1 GB budget) lane 4c's requested
ratio is unreachable and only `runtime.subcall_sizing_binding` reveals it
(`run_cells.sh:143-148` never raises the cap). **Fix:** print a loud warning at
resolve time when `binding != "budget"` and the target ratio was explicit.

---

## D. Cleanup / stale artifacts

### D1. `run_infolab.sh` — P1 (trap) / P2 (dead code)
- **No `HUP` in the trap list** (`EXIT INT TERM` only): an ssh drop orphans the
  `vllm serve` children, which keep the port and ~44 GB until killed by hand (bitten in
  practice; tmux is the current workaround). Add `HUP`.
- `setup` still clones `snu-mllab/KVzip`, installs flash-attn, and builds
  `tiny_api_cuda` (lines ~163-181) — all dead since the standalone-KVzip backend was
  replaced by the kvpress-based `KVzipSubClient`. *(RLM.md §7.7.)* Delete, and drop the
  `.venv` juggling the old backend needed.

### D2. Stale `evaluation/rlm/loft128k/README.md` — P2 *(RLM.md §7.9)*
Still documents physical eviction, `KVZIP_DIR`, `--compression-ratio`,
`.venv-kvpress`. Rewrite against the current backend; `RLM.md` is the source of truth.

### D3. `test_kvzip_direct_smoke.py` — P2 *(RLM.md §7.10)*
`__main__` script with a hardcoded `/home/rethinkingai-self/...` model path, not a
pytest test. Parametrize the model path via env var and either convert to a skippable
pytest (`@pytest.mark.skipif(no GPU)`) or move it out of the package.

### D4. `make test` does not collect `evaluation/rlm/` — P1 *(RLM.md §7.6)*
`test_guards.py`, `test_run_benchmark.py`, `test_kvzip_backend.py` never run in the
default target (the sizing tests were placed under `tests/evaluation/` precisely to
dodge this). **Fix:** move the three files to `tests/evaluation/` (matching the sizing
tests) rather than widening collection — keeps upstream's layout convention.

---

## Keep documented, don't "fix"

- **KVzip compression is logical, not physical** *(RLM.md §7.1-2)*: `KVzipPress` masks
  evicted KV; the full uncompressed cache stays resident. Budget-derived sizing buys
  comparability, not headroom. Any write-up claiming "compression enables bigger reads"
  is about the method, not this implementation.
- **Decimal GB vs binary GiB coexist deliberately** *(RLM.md §7.4)*: budgets in decimal
  to match published matrix numbers, GPU headroom in binary.
- **Auto sizing is not bit-reproducible across machines** *(RLM.md §7.5)*:
  `mem_get_info` is device-global; the 2% resume guard is the mitigation.
- **The venv conflict is structural** *(RLM.md §7.8)*: kvpress wants
  `transformers>=4.56` while `.venv` pins `4.51.3` for vLLM 0.8.5
  (`all_special_tokens_extended` removal). Two venvs remain the answer until the
  vLLM pin can move; D1's cleanup only removes the *dead third* environment.
- **`ungrounded_final` in the gen-1 LOFT numbers is honest**: the rejected abstentions
  would have scored 0 anyway; `max_steps`/`run_timeout` totals were negligible. B1 is
  about semantics and triage, not about recovering score — do **not** "fix" it by
  raising `--max-steps` or `--run-timeout`.

---

## What landed where

| Area | Files |
|---|---|
| Abstention channel, grounding rework, overflow retry, sub-call cap, ports A1–A5 | `evaluation/rlm/rlm.py` |
| Server-reported prompt tokens | `evaluation/rlm/client.py` |
| De-dup, resume guard, `--split`, `answer_format`, abstention + cost-axis metrics | `evaluation/rlm/run_benchmark.py` |
| Split filter, dev-only-`--limit` warning, `split` on each example | `evaluation/benchmarks/loaders.py`, `evaluation/rlm/datasets.py` |
| Scratchpad label, per-dataset headline key, root/sub cost columns | `evaluation/compare.py` |
| One results-path definition | `evaluation/results_layout.py` (new), `score.py`, `compare.py`, `run_benchmark.py` |
| System prompt off the compressed side | `evaluation/rlm/kvzip_backend.py` |
| `HUP` trap, dead KVzip setup removed | `evaluation/rlm/loft128k/run_infolab.sh` |
| Tests moved into `make test`'s reach, plus new coverage | `tests/evaluation/test_rlm_{guards,run_benchmark,subcall_context}.py`, `tests/evaluation/test_kvzip_backend.py` |
| Smoke script un-hardcoded, out of the pytest namespace | `evaluation/rlm/kvzip_direct_smoke.py` |
| Docs re-baselined | `RLM.md` §7, `evaluation/rlm/README.md`, `evaluation/rlm/loft128k/README.md`, `QWEN3_LOCAL_RUNBOOK.md` |

Two decisions worth flagging, both departures from what this document originally
proposed:

- **`sub_model` is NOT in the run-dir slug.** Adding it would rename every run
  directory ever written and silently orphan their checkpoints. It is enforced on
  resume instead, along with `--limit`, `--split`, `--max-steps`, the timeouts and the
  sub-call caps: `config.yaml` is now written *before* the first example runs, so a
  resume always has something to check against. Erroring loudly beats renaming quietly.
- **`_mentions` does not rescue every multiple-choice task.** Word boundaries stop a
  literal from matching *inside* a longer token (a question mentioning 2012 no longer
  licenses the invented answer 12), but a task that enumerates `(A) ... (B) ...` still
  grounds a bare `B`, because the label really is present in the task text. String
  matching cannot separate "mentioned as an option" from "mentioned as the answer";
  the abstention channel and the AST tightening are what actually reduce that surface.

When landing upstream: SPDX header on every `.py`, DCO sign-off (`git commit -s`),
`🤖🤖🤖` appended to commit messages and PR titles, `make style` + `make test`
(black/isort line length 120). On the Mac, never run `make format`/`make style`
directly — use `.venv/bin/python -m black` / `.venv-mac/bin/python -m pytest` (the
`uv run` sync corrupts the venv). Note `make style` was already failing on master for
unrelated files (`evaluation/benchmarks/synthetickv*/generate_dataset.py`); the files
changed here are clean.

