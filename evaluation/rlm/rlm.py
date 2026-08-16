# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recursive Language Model (RLM) with a configurable MEMORY BUDGET (eviction-only).

This is a drop-in successor to the minimal scaffold. The paradigm is unchanged
(Zhang, Kraska & Khattab, 2025): the long prompt is NOT placed in the model's
context window; it lives as a string `context` in a persistent REPL, and the
root LM writes code to inspect it / recurse via `llm_query`.

MemoryBudget caps the ROOT model's context window (in tokens), independent of
document length. The full transcript is kept server-side, but the root only ever
SEES a bounded view: system + begin + the most-recent turns that fit the budget.
Older turns are simply DROPPED (evicted) — there is no scratchpad and no
summarization. Because REPL variables persist across turns, the model can always
recompute or re-fetch anything an evicted turn produced.

Grounding note: the anti-hallucination guard (FINAL literal must appear in real
output) checks a SEPARATE server-side `seen_output` accumulator, which is never
compacted. So shrinking the budget never weakens grounding.

NOTE: model-generated code is executed with `exec`. On a shared cluster, run this
inside your own user account / a SLURM job only. For stricter isolation, wrap the
REPL in a separate process or container.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

from .client import NIMClient

ROOT_SYSTEM_PROMPT = """You are solving a task over a LONG document that does NOT fit in your context window.
The document is stored in a Python REPL as a string variable named `context`
(length: {ctx_len} characters). You interact with it by writing Python code.

RULES:
- Reply with exactly ONE Python code block, fenced as ```python ... ```, and then
  STOP. Write NOTHING after the code block.
- CRITICAL: You CANNOT see the result of your code until the next message. NEVER
  write, guess, or simulate the REPL output yourself. NEVER state an answer you
  have not literally seen in a real REPL output message.
- The REPL is persistent: variables survive across turns. Like Python's
  interactive shell, the value of a bare final expression is echoed back to you.
- Useful tools available in the REPL:
    * `context` (str): the full document.
    * `llm_query(prompt: str) -> str`: ask a sub-LLM about text. IMPORTANT: the
      sub-LLM CANNOT see `context` or any of your variables — it sees ONLY the
      prompt string you pass. You MUST embed the actual text snippet inside the
      prompt, e.g. llm_query("Answer X based on this text:\\n" + context[i:j])
      (keep each call under ~8000 characters). Capture the result:
      ans = llm_query(...) then print(ans).
    * `print(...)`: anything you print is shown back to you in the NEXT message
      (truncated to {obs_limit} chars), so print only what you need to see.{note_tool}
- Strategy: peek at structure first (e.g. `print(context[:2000])`,
  `print(len(context))`, regex search), then narrow down with string ops or
  chunked `llm_query` calls. Do NOT print the whole context.
- If a search or extraction returns ZERO matches or fewer than the task implies,
  that is a signal your pattern is wrong — NOT that the answer is 0 or empty.
    Print a sample of the text around a likely keyword to see the actual format,
      then retry with a corrected pattern.
      - When the task states how many items exist, verify your extraction found that
        many before computing the final answer.
- Once (and only once) you have SEEN the information needed for the answer in a
  real REPL output, finish with a code block calling FINAL(...) on a VARIABLE or
  expression computed by your code. NEVER call FINAL with a literal guessed
  value — FINAL("42") with a made-up constant will be rejected.
- You have at most {max_steps} code turns. Be efficient.{budget_note}

Example of a correct session (3 turns):
  You:  ```python
        idx = context.find("invoice total")
        print(context[max(0, idx-200):idx+500])
        ```
  REPL: ...The invoice total for March was $4,210 including tax...
  You:  ```python
        ans = llm_query("What was the March invoice total? Answer from this text:\\n" + context[idx-200:idx+500])
        print(ans)
        ```
  REPL: $4,210
  You:  ```python
        FINAL(ans)
        ```

The user's task is:
{task}
"""

SUB_SYSTEM_PROMPT = (
    "You are a helpful sub-model. Answer the question using ONLY the text "
    "provided in the prompt. Be concise and factual."
)

FOLD_MARKER = (
    "[MEMORY NOTICE] {n} earlier turn(s) were dropped to stay within your memory "
    "budget. They are gone from your context and were NOT saved anywhere — but the "
    "REPL is persistent, so any Python variable you assigned still exists. Re-query "
    "the REPL if you need anything from those turns. Continue from the recent output "
    "below."
)

NOTE_TOOL_HELP = (
    "\n    * `note(text: str)`: save a SHORT finding to a persistent scratchpad. "
    "The scratchpad is shown back to you on every turn, so notes never scroll "
    "out of view — save key facts, indices, and partial answers there."
)

SCRATCHPAD_TEMPLATE = "[SCRATCHPAD] Your saved notes so far (persistent — always visible):\n{notes}\n[END SCRATCHPAD]"

FOLD_SCRATCHPAD_TEMPLATE = (
    "[MEMORY NOTICE] {n} earlier turn(s) were dropped to stay within your memory "
    "budget. Raw output from them is gone, but the REPL is persistent (your "
    "variables still exist) and this scratchpad survived:\n"
    "{notes}\n[END SCRATCHPAD] Continue from the recent output below."
)


class _ExecTimeout(Exception):
    """Raised by the SIGALRM handler when model-generated code exceeds its budget."""


def _raise_exec_timeout(signum, frame):  # pragma: no cover - signal handler
    raise _ExecTimeout()


CODE_RE = re.compile(r"```(?:python|repl|py)?[ \t]*\n?(.*?)```", re.DOTALL)
STMT_KEYWORDS = r"print|import|from|for|while|if|elif|try|except|finally|with|return|FINAL_VAR|FINAL"
ONELINE_FIX_RE = re.compile(rf"(?<=[\)\w'\"])\s+(?=(?:{STMT_KEYWORDS})\b)")
TEXT_FINAL_RE = re.compile(
    r"FINAL\(\s*(?:\"\"\"|'''|\"|')(.*?)(?:\"\"\"|'''|\"|')\s*\)", re.DOTALL
)
TEXT_FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*[\"'](\w+)[\"']\s*\)")


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
class TokenCounter:
    """Token counter: uses tiktoken cl100k_base if installed, else len//4."""

    def __init__(self, fn: Optional[Callable[[str], int]] = None):
        self._fn = fn
        self._enc = None
        if fn is None:
            try:  # pragma: no cover - depends on environment
                import tiktoken

                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None

    def count(self, text: str) -> int:
        if self._fn is not None:
            return self._fn(text)
        if self._enc is not None:
            try:  # pragma: no cover
                return len(self._enc.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict]) -> int:
        # +4 tokens/message is the usual chat-format overhead approximation.
        return sum(self.count(m.get("content", "")) + 4 for m in messages)


# --------------------------------------------------------------------------- #
# Memory budget
# --------------------------------------------------------------------------- #
@dataclass
class MemoryBudget:
    """Caps the ROOT model's context window (eviction-only).

    max_context_tokens : hard cap on the root prompt (system + begin + recent
                         turns). The headline knob to sweep.
    keep_recent_turns  : how many most-recent (assistant,observation) pairs to
                         try to keep verbatim before budget squeezing kicks in.

    Older turns that don't fit are simply DROPPED (evicted) — nothing is saved or
    summarized. The model must keep what it needs in persistent REPL variables.
    """

    max_context_tokens: int = 4096
    keep_recent_turns: int = 3


@dataclass
class Scratchpad:
    """Opt-in persistent notes for the root model (orthogonal to MemoryBudget).

    Enables a `note(text)` REPL tool; saved notes are re-injected into the
    root's view every turn (and survive budget eviction when a MemoryBudget is
    also set).

    max_notes_tokens : cap on the notes block injected into context; oldest
                       text is truncated away once exceeded.
    """

    max_notes_tokens: int = 1024


@dataclass
class RLMResult:
    answer: str | None
    steps: int
    finished: bool
    transcript: list = field(default_factory=list)
    end_reason: str = ""
    metrics: dict = field(default_factory=dict)


class RLM:
    def __init__(
        self,
        root_client: NIMClient,
        sub_client: NIMClient | None = None,
        max_steps: int = 12,
        obs_limit: int = 6000,
        max_subcall_chars: int = 32000,
        budget: Optional[MemoryBudget] = None,
        scratchpad: Optional[Scratchpad] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        cache_subcalls: bool = True,
        exec_timeout: Optional[float] = 60.0,
        run_timeout: Optional[float] = 900.0,
        max_sub_calls: Optional[int] = 40,
    ):
        self.root = root_client
        self.sub = sub_client or root_client
        self.max_steps = max_steps
        self.obs_limit = obs_limit
        self.max_subcall_chars = max_subcall_chars
        self.budget = budget
        self.scratchpad = scratchpad
        self.cache_subcalls = cache_subcalls
        # Each guard below is independently disabled by passing None (or 0 from the
        # CLI). They are on by default because an unguarded run can consume a SLURM
        # allocation on one pathological example -- but a debugging session that
        # wants to watch a single example run to completion can switch any of them
        # off without affecting the others.
        self.exec_timeout = exec_timeout
        # Wall-clock ceiling for ONE example. exec_timeout deliberately excludes
        # llm_query time (a slow sub-call is legitimate), and NIMClient retries with
        # backoff -- so without a separate deadline a single pathological example can
        # stall a job for hours. None disables.
        self.run_timeout = run_timeout
        self.max_sub_calls = max_sub_calls
        self._sub_call_budget: Optional[int] = None
        self._deadline: Optional[float] = None
        self.tok = TokenCounter(token_counter)
        # True only while _exec's SIGALRM code-timeout is armed; lets llm_query
        # pause that watchdog around its (legit, possibly slow) sub-LLM call.
        self._alarm_active = False

    # ---------------- REPL plumbing ----------------
    def _make_env(self, context: str, metrics: dict, cache: dict, notes: list) -> dict:
        final_box: dict = {"value": None, "done": False}

        def llm_query(prompt: str) -> str:
            # Key on the FULL prompt. Keying on the TRUNCATED one makes two different
            # calls that share a 32k-char prefix collide -- e.g. context[0:100000] and
            # context[0:200000] -- silently returning one slice's answer for the other.
            key = str(prompt)
            if self.cache_subcalls and key in cache:
                metrics["sub_cache_hits"] += 1
                return cache[key]
            if self._sub_call_budget is not None and metrics["sub_calls"] >= self._sub_call_budget:
                return (
                    "[SUB-CALL LIMIT REACHED] No further llm_query calls are "
                    "available for this example. Answer from what you have already "
                    "seen, or use plain string/regex operations on `context`."
                )
            prompt = key[: self.max_subcall_chars]
            if len(key) > self.max_subcall_chars:
                prompt += (
                    f"\n[NOTE: your prompt was truncated at "
                    f"{self.max_subcall_chars} chars; pass a smaller snippet]"
                )
            # Pause the code-exec watchdog (armed in _exec) around this blocking
            # sub-LLM call: a slow-but-legit network call / rate-limit sleep must
            # NOT be mistaken for a runaway loop. Only pure-Python time between
            # sub-calls counts toward exec_timeout.
            remaining = None
            if self._alarm_active:
                remaining, _ = signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                ans = self.sub.chat(
                    [
                        {"role": "system", "content": SUB_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
            finally:
                # `remaining is not None`, not `remaining` -- setitimer returns 0.0
                # for an already-expired timer, and a truthiness test then skips
                # re-arming, silently disabling the exec watchdog for the REST of the
                # cell while _alarm_active still reads True.
                if self._alarm_active and remaining is not None:
                    signal.setitimer(signal.ITIMER_REAL, max(remaining, 0.05))
            metrics["sub_calls"] += 1
            metrics["sub_call_tokens"] += self.tok.count(prompt) + self.tok.count(ans)
            if self.cache_subcalls:
                cache[key] = ans
            return ans

        def FINAL(answer) -> None:
            final_box["value"] = str(answer)
            final_box["done"] = True

        env = {
            "context": context,
            "llm_query": llm_query,
            "FINAL": FINAL,
            "re": re,
        }

        if self.scratchpad is not None:

            def note(text) -> str:
                text = str(text).strip()
                if text:
                    notes.append(text)
                    metrics["notes_saved"] += 1
                return f"[saved note #{len(notes)}]"

            env["note"] = note

        def FINAL_VAR(name) -> None:
            final_box["value"] = str(env.get(str(name), f"<missing var {name}>"))
            final_box["done"] = True

        env["FINAL_VAR"] = FINAL_VAR
        env["_final_box"] = final_box
        return env

    def _exec(self, code: str, env: dict, obs_limit: int) -> str:
        code = code.strip()
        note = ""
        try:
            compile(code, "<rlm>", "exec")
        except SyntaxError:
            if "\n" not in code:
                fixed = ONELINE_FIX_RE.sub("\n", code)
                try:
                    compile(fixed, "<rlm>", "exec")
                    code = fixed
                    note = (
                        "\n[note: your code block was written on a single line; it was "
                        "auto-reformatted. Please use real newlines inside code blocks.]"
                    )
                except SyntaxError:
                    return (
                        "[SYNTAX ERROR] Your code block was written on a single line and "
                        "could not be parsed. Rewrite it as a properly formatted multi-line "
                        "```python block with real newlines."
                    )
        buf = io.StringIO()
        # Bound model-generated code with a wall-clock timeout so an infinite or
        # runaway loop can't hang the whole benchmark. SIGALRM interrupts a stuck
        # pure-Python loop between bytecodes (main thread only); it does NOT
        # interrupt a C-level regex — a known CPython limitation.
        use_alarm = (
            bool(self.exec_timeout)
            and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )
        old_handler = None
        try:
            with contextlib.redirect_stdout(buf):
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    tree = None
                literals = []
                if tree:
                    for node in ast.walk(tree):
                        if (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id == "FINAL"
                            and node.args
                            and isinstance(node.args[0], ast.Constant)
                        ):
                            literals.append(str(node.args[0].value))
                env["_rlm_final_literals"] = literals
                if use_alarm:
                    old_handler = signal.signal(signal.SIGALRM, _raise_exec_timeout)
                    signal.setitimer(signal.ITIMER_REAL, self.exec_timeout)
                    self._alarm_active = True
                if tree and tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = tree.body[-1]
                    assign = ast.Assign(
                        targets=[ast.Name(id="_rlm_last", ctx=ast.Store())],
                        value=last.value,
                    )
                    tree.body[-1] = ast.copy_location(assign, last)
                    ast.fix_missing_locations(tree)
                    sentinel = object()
                    env["_rlm_last"] = sentinel
                    exec(compile(tree, "<rlm>", "exec"), env)  # noqa: S102
                    val = env.get("_rlm_last")
                    if val is not sentinel and val is not None:
                        env["_"] = val
                        print(val if isinstance(val, str) else repr(val))
                else:
                    exec(code, env)  # noqa: S102
        except _ExecTimeout:
            buf.write(
                f"\n[TIMEOUT] your code ran longer than {self.exec_timeout:.0f}s and was "
                "aborted — almost certainly an infinite or runaway loop. Rewrite it to "
                "terminate: avoid unbounded while-loops, bound every iteration, and operate "
                "on slices of `context` instead of rescanning it repeatedly."
            )
        except KeyboardInterrupt:
            raise  # operator Ctrl-C must still stop the run
        except BaseException:
            # BaseException, not Exception: model code calling exit()/quit()/sys.exit()
            # raises SystemExit, which would otherwise escape _exec, escape run(), and
            # escape run_benchmark's `except Exception` -- terminating the whole sweep
            # mid-campaign. Treat it as an ordinary cell failure.
            buf.write("\n[EXCEPTION]\n" + traceback.format_exc(limit=3))
        finally:
            self._alarm_active = False
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
        out = buf.getvalue()
        if len(out) > obs_limit:
            half = obs_limit // 2
            out = (
                out[:half]
                + f"\n...[truncated {len(out) - obs_limit} chars]...\n"
                + out[-half:]
            )
        return (out if out.strip() else "[no output]") + note

    # ---------------- budget / context view ----------------
    def _budget_note(self) -> str:
        if self.budget is None:
            return ""
        if self.scratchpad is not None:
            return (
                f"\n- MEMORY BUDGET: your working context is capped at ~{self.budget.max_context_tokens} "
                "tokens. Once you exceed it, your OLDEST turns are DROPPED automatically. Raw REPL "
                "output that scrolls out vanishes, but Python VARIABLES persist across turns and "
                "your note() SCRATCHPAD is always re-shown. So save anything you will need for "
                "FINAL with note('...') (or keep it in a variable); never rely on old output "
                "staying visible."
            )
        return (
            f"\n- MEMORY BUDGET: your working context is capped at ~{self.budget.max_context_tokens} "
            "tokens. Once you exceed it, your OLDEST turns are DROPPED automatically and are gone "
            "for good — they are not saved or summarized anywhere. Raw REPL output that scrolls out "
            "vanishes, but Python VARIABLES in the REPL persist across turns. So store anything you "
            "will need for FINAL in a variable (or be ready to recompute it); never rely on old "
            "output staying visible."
        )

    def _notes_block(self, notes: list) -> str:
        text = "\n".join(f"- {n}" for n in notes) if notes else "(nothing saved yet)"
        cap = self.scratchpad.max_notes_tokens if self.scratchpad else 1024
        # keep the scratchpad itself within its sub-budget (drop OLDEST text first)
        while self.tok.count(text) > cap and len(text) > 200:
            text = "- ...[oldest notes truncated]\n" + text[int(len(text) * 0.2) :]
        return text

    # ---------------- main loop ----------------
    def run(self, context: str, task: str) -> RLMResult:
        metrics = {
            "steps": 0,
            "root_prompt_tokens": 0,
            "root_completion_tokens": 0,
            "peak_context_tokens": 0,
            "sub_calls": 0,
            "sub_call_tokens": 0,
            "sub_cache_hits": 0,
            "evictions": 0,
            "budget": (self.budget.max_context_tokens if self.budget else None),
        }
        if self.scratchpad is not None:
            metrics["scratchpad"] = True
            metrics["notes_saved"] = 0
        cache: dict = {}
        notes: list = []
        self._sub_call_budget = self.max_sub_calls
        self._deadline = time.monotonic() + self.run_timeout if self.run_timeout else None
        env = self._make_env(context, metrics, cache, notes)

        system_msg = {
            "role": "system",
            "content": ROOT_SYSTEM_PROMPT.format(
                ctx_len=len(context),
                obs_limit=self.obs_limit,
                max_steps=self.max_steps,
                task=task,
                budget_note=self._budget_note(),
                note_tool=(NOTE_TOOL_HELP if self.scratchpad is not None else ""),
            ),
        }
        begin_msg = {"role": "user", "content": "Begin. Write your first code block."}

        full_history: list = []  # all (assistant, user) messages, server-side
        evicted_count = 0  # number of leading pairs already dropped (evicted)
        transcript: list = []
        nudges = 0
        last_code_norm: str | None = None  # repetition-breaker state (see below)
        repeat_count = 0
        seen_output = ""  # grounding accumulator — NEVER compacted

        def build_sent() -> tuple[list, int]:
            """Construct the bounded view actually sent to the root model."""
            nonlocal evicted_count
            base = [system_msg, begin_msg]
            if self.budget is None:
                # Scratchpad notes are re-injected every turn even without a
                # budget — that visibility is the whole point of the tool.
                if self.scratchpad is not None and notes:
                    notes_msg = {
                        "role": "user",
                        "content": SCRATCHPAD_TEMPLATE.format(notes=self._notes_block(notes)),
                    }
                    return base + [notes_msg] + full_history, 0
                return base + full_history, 0

            n_pairs = len(full_history) // 2
            pairs = [full_history[i * 2 : i * 2 + 2] for i in range(n_pairs)]
            keep = self.budget.keep_recent_turns
            kept = pairs[-keep:] if keep > 0 else []

            def assemble(kept_pairs, fold_n):
                msgs = list(base)
                if self.scratchpad is not None:
                    if fold_n > 0:
                        msgs.append(
                            {
                                "role": "user",
                                "content": FOLD_SCRATCHPAD_TEMPLATE.format(n=fold_n, notes=self._notes_block(notes)),
                            }
                        )
                    elif notes:
                        msgs.append(
                            {
                                "role": "user",
                                "content": SCRATCHPAD_TEMPLATE.format(notes=self._notes_block(notes)),
                            }
                        )
                elif fold_n > 0:
                    msgs.append(
                        {"role": "user", "content": FOLD_MARKER.format(n=fold_n)}
                    )
                for p in kept_pairs:
                    msgs.extend(p)
                return msgs

            fold_n = n_pairs - len(kept)
            sent = assemble(kept, fold_n)
            # squeeze: drop more recent pairs until within budget
            while (
                self.tok.count_messages(sent) > self.budget.max_context_tokens and kept
            ):
                kept = kept[1:]
                fold_n = n_pairs - len(kept)
                sent = assemble(kept, fold_n)

            # record newly-evicted pairs (dropped for good)
            if fold_n > evicted_count:
                evicted_count = fold_n
                metrics["evictions"] += 1
            return sent, fold_n

        for step in range(1, self.max_steps + 1):
            if self._deadline is not None and time.monotonic() > self._deadline:
                # Out of wall-clock. A distinct end_reason so a timeout stays
                # distinguishable from a genuine abstention when triaging.
                return RLMResult(None, step, False, transcript, "run_timeout", metrics)
            metrics["steps"] = step
            sent, _ = build_sent()
            ctx_tokens = self.tok.count_messages(sent)
            metrics["root_prompt_tokens"] += ctx_tokens
            metrics["peak_context_tokens"] = max(
                metrics["peak_context_tokens"], ctx_tokens
            )

            reply = self.root.chat(sent)
            metrics["root_completion_tokens"] += self.tok.count(reply)
            blocks = CODE_RE.findall(reply)

            # adaptive observation limit: never let one observation exceed the budget
            obs_limit = self.obs_limit
            if self.budget is not None:
                obs_limit = max(
                    512, min(self.obs_limit, self.budget.max_context_tokens * 3)
                )

            # --- No code block in the reply ---
            if not blocks:
                m = TEXT_FINAL_RE.search(reply)
                if m:
                    val = m.group(1).strip()
                    if val and (val in seen_output or val in task):
                        transcript.append(
                            {
                                "step": step,
                                "reply": reply,
                                "code": None,
                                "observation": "[FINAL parsed from prose]",
                            }
                        )
                        return RLMResult(
                            val, step, True, transcript, "final_in_prose", metrics
                        )
                    nudges += 1
                    transcript.append(
                        {
                            "step": step,
                            "reply": reply,
                            "code": None,
                            "observation": "[ungrounded prose FINAL rejected]",
                        }
                    )
                    if nudges > 2:
                        return RLMResult(
                            None, step, False, transcript, "ungrounded_final", metrics
                        )
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append(
                        {
                            "role": "user",
                            "content": f"REJECTED: your answer {val!r} never appeared in any actual REPL "
                            "output, so it looks like a guess. Do NOT invent answers. Write a "
                            "```python code block that finds the answer in `context` (string "
                            "search / regex / llm_query with the snippet pasted in), look at "
                            "the real output, and only then FINAL it.",
                        }
                    )
                    continue
                mv = TEXT_FINAL_VAR_RE.search(reply)
                if mv:
                    val = str(env.get(mv.group(1), f"<missing var {mv.group(1)}>"))
                    transcript.append(
                        {
                            "step": step,
                            "reply": reply,
                            "code": None,
                            "observation": "[FINAL_VAR parsed from prose]",
                        }
                    )
                    return RLMResult(
                        val, step, True, transcript, "final_var_in_prose", metrics
                    )
                nudges += 1
                transcript.append(
                    {
                        "step": step,
                        "reply": reply,
                        "code": None,
                        "observation": "[no code block - nudged]",
                    }
                )
                if nudges > 2:
                    return RLMResult(
                        reply.strip(),
                        step,
                        False,
                        transcript,
                        "gave_up_no_code",
                        metrics,
                    )
                full_history.append({"role": "assistant", "content": reply})
                full_history.append(
                    {
                        "role": "user",
                        "content": "Your reply contained no code block, so NOTHING was executed and no "
                        "answer was recorded. Reply with exactly one ```python code block. "
                        "If you already know the answer from a previous REPL output, reply with "
                        'a code block containing only: FINAL("your answer")',
                    }
                )
                continue

            # --- Execute ONLY the first block ---
            code = blocks[0]
            obs = self._exec(code, env, obs_limit)
            if len(blocks) > 1:
                obs += (
                    "\n[WARNING: you wrote multiple code blocks; ONLY the FIRST was "
                    "executed. Anything you wrote after it (including any 'output' you "
                    "predicted) did NOT happen.)"
                )
            transcript.append(
                {"step": step, "reply": reply, "code": code, "observation": obs}
            )
            seen_output += "\n" + obs

            if env["_final_box"]["done"]:
                val = env["_final_box"]["value"]
                if (
                    val in (env.get("_rlm_final_literals") or [])
                    and val not in seen_output
                    and val not in task
                ):
                    env["_final_box"]["done"] = False
                    env["_final_box"]["value"] = None
                    nudges += 1
                    if nudges > 2:
                        return RLMResult(
                            None, step, False, transcript, "ungrounded_final", metrics
                        )
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append(
                        {
                            "role": "user",
                            "content": f"REJECTED: FINAL({val!r}) is a literal constant that never appeared "
                            "in any REPL output — it looks like a guess. Find the real answer in "
                            "`context` first (e.g. re.search / context.find / llm_query with the "
                            "snippet pasted in), print it, then call FINAL on the variable "
                            "holding it.",
                        }
                    )
                    continue
                return RLMResult(val, step, True, transcript, "final_called", metrics)

            # --- Repetition breaker ---
            # A stuck small model re-emits the SAME code cell every turn — identical
            # code -> identical observation — and burns all max_steps (and the whole
            # run_timeout) without ever calling FINAL. Detect an exact consecutive
            # repeat and escalate: first a hard nudge to stop and answer or change
            # tactic, then a forced GROUNDED fallback that answers from what was
            # actually seen, so the example still yields a real answer instead of
            # grinding to max_steps with score 0.
            code_norm = " ".join(code.split())
            if code_norm == last_code_norm:
                repeat_count += 1
            else:
                repeat_count = 1
                last_code_norm = code_norm

            if repeat_count >= 3:
                # Still looping after the nudge: answer from the real REPL output
                # accumulated so far (grounded), plus the task. Bypasses FINAL()'s
                # literal-grounding guard deliberately — this is the escape hatch.
                material = seen_output.strip()[-self.max_subcall_chars :]
                fb_prompt = (
                    "Answer the task using ONLY the material below, which was extracted "
                    "from a long document by prior code. Be concise and factual.\n\n"
                    f"Task: {task}\n\nExtracted material:\n{material}"
                )
                ans = self.sub.chat(
                    [
                        {"role": "system", "content": SUB_SYSTEM_PROMPT},
                        {"role": "user", "content": fb_prompt},
                    ]
                )
                metrics["sub_calls"] += 1
                metrics["sub_call_tokens"] += self.tok.count(fb_prompt) + self.tok.count(ans)
                transcript.append(
                    {
                        "step": step,
                        "reply": "[repetition-breaker fallback]",
                        "code": None,
                        "observation": "[forced grounded answer after 3 identical code cells]",
                    }
                )
                return RLMResult(ans, step, True, transcript, "repetition_broken", metrics)

            full_history.append({"role": "assistant", "content": reply})
            if repeat_count == 2:
                full_history.append(
                    {
                        "role": "user",
                        "content": "STOP: you just ran this EXACT same code twice and got the same "
                        "result — repeating it will not help. Do ONE of these, and do NOT "
                        "repeat the previous code:\n"
                        "1) If you already have enough to answer, reply with a code block "
                        "containing only FINAL(<expr>) (a variable/expression you computed, "
                        "not a guessed literal).\n"
                        "2) If your search found nothing, your approach is WRONG — the answer "
                        "may be a category/label, or phrased differently. Try a COMPLETELY "
                        "different tactic: inspect context[:2000], search a different keyword, "
                        "or llm_query a relevant snippet.",
                    }
                )
            else:
                full_history.append(
                    {
                        "role": "user",
                        "content": f"ACTUAL REPL output:\n```\n{obs}\n```\n"
                        f"Continue. ({self.max_steps - step} turns left) "
                        "Remember: one code block only; finish with FINAL(...) once you have "
                        "seen the answer in a real output.",
                    }
                )

        return RLMResult(None, self.max_steps, False, transcript, "max_steps", metrics)


def _is_context_overflow(exc: Exception) -> bool:
    """True for a 400 that means 'prompt too long', not some other bad request.

    Matched on message text because the OpenAI client surfaces vLLM's error as a
    generic BadRequestError; vLLM's wording is
    "This model's maximum context length is 40960 tokens...".
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    msg = str(exc).lower()
    return "context length" in msg or "context_length" in msg or "too long" in msg


def vanilla_answer(
    client: NIMClient,
    context: str,
    task: str,
    char_limit: int = 400_000,
    max_prompt_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    stats: Optional[dict] = None,
) -> str:
    """Baseline: stuff (possibly truncated) context directly into the prompt.

    `stats`, if given, is filled with how much context actually survived
    truncation. The caller needs this to separate "the model got it wrong" from
    "the answer was never in the prompt". Without it the vanilla arm's score is
    not comparable to a KVPress run, which always sees the full context: on a
    200k-char synthetic with a 100k limit vanilla is structurally capped, and a
    bare score column reads that cap as model accuracy.
    """
    truncated = context[:char_limit]

    def build(body: str) -> str:
        note = "" if len(context) <= len(body) else "\n[NOTE: document truncated]"
        return f"Document:\n{body}{note}\n\nTask: {task}\nAnswer concisely."

    prompt = build(truncated)

    # A CHARACTER limit is not a token limit. `--vanilla-char-limit 100000` assumes
    # ~4 chars/token, but dense subsets tokenize far tighter -- RULER's `cwe`
    # (repeated word lists) and `niah_multikey_3` (UUID-like keys) run ~2.5, so 100k
    # chars is ~40k tokens and the server rejected EVERY request with a 400
    # "maximum context length is 40960 tokens". Those cells scored 0.0 for vanilla
    # from a harness error rather than from the model. Shrink until it really fits.
    if max_prompt_tokens:
        tok = TokenCounter(token_counter)
        while tok.count(prompt) > max_prompt_tokens and len(truncated) > 2000:
            truncated = truncated[: int(len(truncated) * 0.8)]
            prompt = build(truncated)

    # ...but the PREDICTIVE shrink above cannot be trusted on its own, because the
    # counter is not the server's tokenizer. TokenCounter uses tiktoken cl100k_base
    # when installed and `len // 4` when not -- and `len // 4` is exactly the 4
    # chars/token assumption this whole block exists to escape, so on a box without
    # tiktoken the loop is a silent no-op. Even with tiktoken, cl100k and Qwen3's
    # BPE disagree by well over the available headroom on the dense subsets.
    #
    # So treat the estimate as a first guess and let the SERVER be the authority:
    # shrink and retry whenever it rejects the prompt for length. This is tokenizer-
    # independent and cannot silently no-op.
    def record(retries: int) -> None:
        if stats is not None:
            stats.update(
                context_chars=len(context),
                context_chars_used=len(truncated),
                truncated=len(truncated) < len(context),
                shrink_retries=retries,
            )

    for attempt in range(12):
        try:
            out = client.chat([{"role": "user", "content": prompt}])
            record(attempt)
            return out
        except Exception as e:  # noqa: BLE001 - narrowed by the guard below
            if not _is_context_overflow(e) or len(truncated) <= 2000:
                record(attempt)
                raise
            truncated = truncated[: int(len(truncated) * 0.8)]
            prompt = build(truncated)
    record(12)
    return client.chat([{"role": "user", "content": prompt}])
