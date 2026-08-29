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
Older turns are simply DROPPED (evicted) — nothing is summarized. Because REPL
variables persist across turns, the model can always recompute or re-fetch
anything an evicted turn produced. An opt-in Scratchpad adds a `note()` tool
whose contents ARE re-injected every turn and so survive eviction.

Without a MemoryBudget the view is unbounded, and a long example can outgrow the
served window. The root path therefore treats the SERVER as the authority on what
fits, exactly as `vanilla_answer` does: an overflow 400 evicts the oldest turns
and retries rather than ending the example as an exception.

Grounding note: the anti-hallucination guard (FINAL literal must appear in real
output) checks a SEPARATE server-side `seen_output` accumulator, which is never
compacted and never truncated. So shrinking the budget — or truncating an
observation for display — never weakens grounding. A model that has genuinely
searched and found nothing ends the example with FINAL_NONE(reason), which is a
legitimate answer, not a rejected guess.

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

from .client import LLMClient

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
{llm_query_help}
    * `print(...)`: anything you print is shown back to you in the NEXT message
      (truncated to {obs_limit} chars), so print only what you need to see.{note_tool}
- Strategy: peek at structure first (e.g. `print(context[:2000])`,
  `print(len(context))`, regex search), then narrow down with string ops or
  chunked `llm_query` calls. Do NOT print the whole context.
- If a search or extraction returns ZERO matches or fewer than the task implies,
  that is USUALLY a signal your pattern is wrong — not that the answer is empty.
  Print a sample of the text around a likely keyword to see the actual format,
  then retry with a corrected pattern.
  - When the task states how many items exist, verify your extraction found that
    many before computing the final answer.
- Once (and only once) you have SEEN the information needed for the answer in a
  real REPL output, finish with a code block calling FINAL(...) on a VARIABLE or
  expression computed by your code. NEVER call FINAL with a literal guessed
  value — FINAL("42") with a made-up constant will be rejected.
- If, after genuinely searching, the document does NOT contain the answer, end
  with FINAL_NONE("short reason") instead of guessing. Abstaining is a legitimate
  ending and is always better than inventing a value. Do NOT abstain on your first
  empty search — try at least one different pattern or tactic first.
- You have at most {max_steps} code turns. Be efficient.{budget_note}

Example of a correct session (3 turns):
  You:  ```python
        idx = context.find("invoice total")
        print(context[max(0, idx-200):idx+500])
        ```
  REPL: ...The invoice total for March was $4,210 including tax...
  You:  ```python
{example_llm_query}
        print(ans)
        ```
  REPL: $4,210
  You:  ```python
        FINAL(ans)
        ```

The user's task is:
{task}
{answer_format_note}"""

# The llm_query help/example rendered into ROOT_SYSTEM_PROMPT depends on the sub
# backend. The SPLIT variants teach the two-arg form used when the sub client
# compresses the context's KV separately from the question (see kvzip_backend.py).
# Both now render the REAL cap: the dense text used to advertise ~8000 characters
# while enforcement was at 32000, so the root was told to send four times less
# than it was allowed to. (That wording was pinned byte-identical while arms 2/3
# were mid-campaign; that campaign is closed.)
LLM_QUERY_HELP_DENSE = """\
    * `llm_query(prompt: str) -> str`: ask a sub-LLM about text. IMPORTANT: the
      sub-LLM CANNOT see `context` or any of your variables — it sees ONLY the
      prompt string you pass. You MUST embed the actual text snippet inside the
      prompt, e.g. llm_query("Answer X based on this text:\\n" + context[i:j])
      (each call is truncated at {max_chars} characters, so pass a focused
      snippet, not the whole document). Capture the result:
      ans = llm_query(...) then print(ans)."""

EXAMPLE_LLM_QUERY_DENSE = (
    '        ans = llm_query("What was the March invoice total? Answer from this text:\\n"'
    " + context[idx-200:idx+500])"
)

LLM_QUERY_HELP_SPLIT = """\
    * `llm_query(question: str, context_text: str) -> str`: ask a sub-LLM a
      question about a piece of text. IMPORTANT: the sub-LLM CANNOT see
      `context` or any of your variables — it sees ONLY what you pass. Put the
      question in the FIRST argument and the raw document slice in the SECOND,
      e.g. llm_query("What is X?", context[i:j]). The slice is read through a
      compressed attention cache, so big slices are cheap: use up to
      ~{max_chars} characters per call, and prefer FEWER, BIGGER slices over
      many small ones. Capture the result: ans = llm_query(...) then
      print(ans)."""

EXAMPLE_LLM_QUERY_SPLIT = '        ans = llm_query("What was the March invoice total?", context[idx-200:idx+500])'

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
# Fallback for a reply whose code block was cut off at max_tokens: an OPENING fence
# with no closing one. Without this the reply reads as "no code block", burns a nudge,
# and after three such replies the raw truncated text is returned as the answer.
UNTERMINATED_CODE_RE = re.compile(r"```(?:python|repl|py)?[ \t]*\n(.*)\Z", re.DOTALL)
STMT_KEYWORDS = r"print|import|from|for|while|if|elif|try|except|finally|with|return|note|FINAL_NONE|FINAL_VAR|FINAL"
ONELINE_FIX_RE = re.compile(rf"(?<=[\)\w'\"])\s+(?=(?:{STMT_KEYWORDS})\b)")
TEXT_FINAL_RE = re.compile(r"FINAL\(\s*(?:\"\"\"|'''|\"|')(.*?)(?:\"\"\"|'''|\"|')\s*\)", re.DOTALL)
TEXT_FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*[\"'](\w+)[\"']\s*\)")
# FINAL_NONE may be written bare or with a reason, in code or in prose.
TEXT_FINAL_NONE_RE = re.compile(r"FINAL_NONE\(\s*(?:(?:\"\"\"|'''|\"|')(.*?)(?:\"\"\"|'''|\"|'))?\s*\)", re.DOTALL)

# A sub client that cannot fit a slice reports it as an ordinary return string
# (kvzip_backend._generate), which the root is free to ignore. Recognised here so
# repeated failures can end the tool instead of burning the sub-call budget.
SUB_FIT_FAILURE_PREFIX = "[SUB-MODEL ERROR]"
MAX_SUB_FIT_FAILURES = 3

# Seconds the exec watchdog is always given back after a sub-call, even if its
# timer expired mid-call. See the re-arm site in llm_query.
ALARM_REARM_FLOOR_S = 1.0

# How many characters one token is worth, for the two places that must convert
# between the two units without a tokenizer: TokenCounter's no-tiktoken fallback
# and the adaptive observation limit. Shared so the two cannot silently disagree
# (they used 4 and 3 respectively, for no stated reason).
CHARS_PER_TOKEN_ESTIMATE = 4


def _static_string(node: ast.AST) -> Optional[str]:
    """The value of `node` if it is knowable without running any code, else None.

    This is what separates a GUESS from a computed answer for the grounding guard.
    Matching only `ast.Constant` let three spellings of the same guess through --
    FINAL(f"42"), FINAL("4"+"2") and, once folded, any concatenation of literals --
    so the guard rejected the honest spelling and waved past the evasive ones.
    A node mentioning any variable is not static, and is grounded by construction:
    the value came from code that actually ran.
    """
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):  # f-string: static only if nothing is interpolated
        parts = [_static_string(v) for v in node.values]
        return "".join(parts) if all(p is not None for p in parts) else None  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_string(node.left), _static_string(node.right)
        return left + right if left is not None and right is not None else None
    return None


def _mentions(haystack: str, needle: str) -> bool:
    """Word-boundary containment, for the `answer appears in the task text` check.

    Plain `needle in haystack` whitelisted almost every short answer: any question
    containing "12" anywhere licensed the ungrounded answer 12, and every
    classification/multiple-choice task enumerates its labels in the prompt, so that
    whole class of tasks had no grounding at all.
    """
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


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
        return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)

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
        root_client: LLMClient,
        sub_client: LLMClient | None = None,
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
        subcall_deadline_check: bool = False,
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
        # llm_query time (a slow sub-call is legitimate), and LLMClient retries with
        # backoff -- so without a separate deadline a single pathological example can
        # stall a job for hours. None disables.
        self.run_timeout = run_timeout
        self.max_sub_calls = max_sub_calls
        # Check the run_timeout deadline INSIDE llm_query too. Off by default:
        # the HTTP/vLLM arms take their per-step deadline check between turns, and
        # changing that mid-campaign would confound them. The kvzip arm turns it
        # on because one code cell can loop over many multi-minute in-process
        # sub-calls without ever reaching the per-step check.
        self.subcall_deadline_check = subcall_deadline_check
        self._sub_call_budget: Optional[int] = None
        self._deadline: Optional[float] = None
        self.tok = TokenCounter(token_counter)
        # True only while _exec's SIGALRM code-timeout is armed; lets llm_query
        # pause that watchdog around its (legit, possibly slow) sub-LLM call.
        self._alarm_active = False

    # ---------------- REPL plumbing ----------------
    def _cap_subcall(self, question: str, chunk: Optional[str]) -> tuple[str, Optional[str], bool]:
        """Fit (question, slice) into ONE max_subcall_chars budget.

        Capping each side independently let a split call carry ~2x the size the
        prompt advertises, which is what the sub model's KV fit check then rejects
        -- with a string the root is free to ignore. The question is an instruction
        rather than the payload, so it gets a quarter of the budget and the slice
        takes the rest; on the dense path (no slice) the question gets all of it.
        """
        budget = self.max_subcall_chars
        truncated = False
        if chunk is None:
            if len(question) > budget:
                question, truncated = question[:budget], True
            return question, None, truncated
        question_cap = max(1, budget // 4)
        if len(question) > question_cap:
            question, truncated = question[:question_cap], True
        room = max(0, budget - len(question))
        if len(chunk) > room:
            chunk, truncated = chunk[:room], True
        return question, chunk, truncated

    def _make_env(self, context: str, metrics: dict, cache: dict, notes: list) -> dict:
        final_box: dict = {"value": None, "done": False}

        def llm_query(prompt: str, context: str | None = None) -> str:
            # Key on the FULL strings. Keying on the TRUNCATED ones makes two
            # different calls that share a 32k-char prefix collide -- e.g.
            # context[0:100000] and context[0:200000] -- silently returning one
            # slice's answer for the other. The key is a (question, slice) pair so
            # the same question over two slices never collides either.
            question = str(prompt)
            chunk = None if context is None else str(context)
            key = (question, chunk)
            if self.cache_subcalls and key in cache:
                metrics["sub_cache_hits"] += 1
                return cache[key]
            if self._sub_call_budget is not None and metrics["sub_calls"] >= self._sub_call_budget:
                return (
                    "[SUB-CALL LIMIT REACHED] No further llm_query calls are "
                    "available for this example. Answer from what you have already "
                    "seen, or use plain string/regex operations on `context`."
                )
            if self.subcall_deadline_check and self._deadline is not None and time.monotonic() > self._deadline:
                return (
                    "[TIME LIMIT REACHED] The per-example time budget is exhausted; "
                    "no further llm_query calls are available. Answer NOW from what "
                    "you have already seen, or use plain string/regex operations on "
                    "`context`."
                )
            if metrics.get("sub_fit_failures", 0) >= MAX_SUB_FIT_FAILURES:
                # The sub model has refused this many slices for not fitting in its
                # KV budget. It returns that refusal as an ordinary string, so a root
                # that ignores it can burn every remaining sub-call on slices that
                # will never fit. Stop offering the tool instead.
                return (
                    "[SUB-CALL LIMIT REACHED] The sub-model rejected too many slices "
                    "as too large to fit. No further llm_query calls are available. "
                    "Use plain string/regex operations on `context` instead."
                )
            if chunk is not None and not hasattr(self.sub, "chat_split"):
                # Two-arg call against a plain chat backend (arms without a
                # split-capable sub): fold the slice into the prompt so the call
                # still behaves like the documented one-arg form.
                question = f"{question}\n\n{chunk}"
                chunk = None
            question, chunk, truncated = self._cap_subcall(question, chunk)
            if truncated:
                # Always on the QUESTION side: on the split path the context slice
                # is what gets compressed, and a notice buried there can be evicted.
                question += (
                    f"\n[NOTE: your prompt was truncated at {self.max_subcall_chars} "
                    "chars TOTAL; pass a smaller snippet]"
                )
            # Pause the code-exec watchdog (armed in _exec) around this blocking
            # sub-LLM call: a slow-but-legit network call / rate-limit sleep must
            # NOT be mistaken for a runaway loop. Only pure-Python time between
            # sub-calls counts toward exec_timeout.
            remaining = None
            if self._alarm_active:
                remaining, _ = signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                if chunk is not None:
                    # The client seam is duck-typed (hasattr-guarded above), so the
                    # LLMClient annotation legitimately lacks this method.
                    ans = self.sub.chat_split(  # type: ignore[attr-defined]
                        question=question, context=chunk, system=SUB_SYSTEM_PROMPT
                    )
                else:
                    ans = self.sub.chat(
                        [
                            {"role": "system", "content": SUB_SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                        ]
                    )
            finally:
                # `remaining is not None`, not `remaining` -- setitimer returns 0.0
                # for an already-expired timer, and a truthiness test then skips
                # re-arming, silently disabling the exec watchdog for the REST of the
                # cell while _alarm_active still reads True.
                if self._alarm_active and remaining is not None:
                    # Grant a floor rather than re-arming at the literal remainder:
                    # a timer that expired DURING the sub-call would otherwise fire
                    # ~50ms after llm_query returns, billing sub-call latency to the
                    # pure-Python code watchdog and aborting a cell that never looped.
                    signal.setitimer(signal.ITIMER_REAL, max(remaining, ALARM_REARM_FLOOR_S))
            metrics["sub_calls"] += 1
            metrics["sub_call_tokens"] += (
                self.tok.count(question) + (self.tok.count(chunk) if chunk else 0) + self.tok.count(ans)
            )
            if chunk is not None:
                metrics["sub_split_calls"] = metrics.get("sub_split_calls", 0) + 1
            if isinstance(ans, str) and ans.startswith(SUB_FIT_FAILURE_PREFIX):
                # Don't cache a refusal: the same slice may fit once the sub model's
                # neighbours free memory, and caching would make the failure permanent.
                metrics["sub_fit_failures"] = metrics.get("sub_fit_failures", 0) + 1
                return ans
            if self.cache_subcalls:
                cache[key] = ans
            return ans

        def FINAL(answer) -> None:
            final_box["value"] = str(answer)
            final_box["done"] = True

        def FINAL_NONE(reason: str = "") -> None:
            """Abstain: the document does not contain the answer.

            Distinct from FINAL so the grounding guard cannot reject it. A model
            that searched honestly and found nothing used to have no way to say so
            -- FINAL("unknown") is by construction absent from the REPL output, so
            the guard read the one truthful ending as a hallucinated guess.
            """
            final_box["value"] = None
            final_box["reason"] = str(reason).strip()
            final_box["done"] = True
            final_box["abstained"] = True

        env = {
            "context": context,
            "llm_query": llm_query,
            "FINAL": FINAL,
            "FINAL_NONE": FINAL_NONE,
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
            # Raise rather than answering "<missing var X>", which used to be recorded
            # as a FINISHED, successful prediction. The NameError lands in the REPL
            # observation, so the model sees and can correct it.
            key = str(name)
            if key not in env:
                raise NameError(
                    f"FINAL_VAR({key!r}): no variable named {key!r} exists in the "
                    "REPL. Assign it first, or call FINAL(<expression>) directly."
                )
            final_box["value"] = str(env[key])
            final_box["done"] = True

        env["FINAL_VAR"] = FINAL_VAR
        env["_final_box"] = final_box
        return env

    def _exec(self, code: str, env: dict, obs_limit: int) -> tuple[str, str]:
        """Run one code cell. Returns (what the model sees, what actually printed).

        The two differ once output exceeds obs_limit. Only the first is sent back
        to the model; the second feeds the grounding accumulator, which must not
        be truncated -- an answer that appeared in the elided middle of a long
        output is grounded, and reading only the display copy called it a guess.
        """
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
                    unparsable = (
                        "[SYNTAX ERROR] Your code block was written on a single line and "
                        "could not be parsed. Rewrite it as a properly formatted multi-line "
                        "```python block with real newlines."
                    )
                    return unparsable, unparsable
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
                        ):
                            static = _static_string(node.args[0])
                            if static is not None:
                                literals.append(static)
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
        full = buf.getvalue()
        out = full
        if len(out) > obs_limit:
            half = obs_limit // 2
            out = out[:half] + f"\n...[truncated {len(out) - obs_limit} chars]...\n" + out[-half:]
        return (out if out.strip() else "[no output]") + note, full

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
        # Keep the scratchpad within its sub-budget by dropping the OLDEST text.
        # The marker goes on once, at the end: prepending it every pass stacked a
        # copy per iteration and then counted each copy against the same cap. The
        # old loop also stopped at 200 chars, so a small cap was silently ignored.
        marker = "- ...[oldest notes truncated]\n"
        truncated = False
        while text and self.tok.count(text) + (self.tok.count(marker) if truncated else 0) > cap:
            text = text[max(1, int(len(text) * 0.2)) :]
            truncated = True
        return marker + text if truncated else text

    # ---------------- main loop ----------------
    def run(self, context: str, task: str, answer_format: str | None = None) -> RLMResult:
        metrics: dict = {
            "steps": 0,
            "root_prompt_tokens": 0,
            "root_completion_tokens": 0,
            "peak_context_tokens": 0,
            "sub_calls": 0,
            "sub_call_tokens": 0,
            "sub_cache_hits": 0,
            # Pre-initialized like every other counter: incremented only via
            # .get(..., 0) + 1, it was absent from a healthy run's metrics and
            # downstream aggregation had to special-case its absence.
            "sub_fit_failures": 0,
            "evictions": 0,
            "overflow_evictions": 0,
            "budget": (self.budget.max_context_tokens if self.budget else None),
        }
        if self.scratchpad is not None:
            metrics["scratchpad"] = True
            metrics["notes_saved"] = 0
        # A split-capable sub client (kvzip backend) accepts the context slice as
        # a separate argument so the press compresses it apart from the question;
        # the system prompt must teach whichever llm_query form is actually wired.
        split_capable = hasattr(self.sub, "chat_split")
        if split_capable:
            metrics["sub_split_calls"] = 0
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
                # The same output contract vanilla_answer receives, phrased for the
                # FINAL() protocol, so neither arm is scored on an answer shape the
                # other was never asked for. Without it the RLM answered via
                # str(FINAL(x)) while vanilla was told "answer concisely", and
                # string-matching metrics measured that style difference as accuracy.
                answer_format_note=(
                    f"\nThe value you pass to FINAL(...) must be formatted as: {answer_format}\n"
                    if answer_format
                    else ""
                ),
                budget_note=self._budget_note(),
                note_tool=(NOTE_TOOL_HELP if self.scratchpad is not None else ""),
                llm_query_help=(LLM_QUERY_HELP_SPLIT if split_capable else LLM_QUERY_HELP_DENSE).format(
                    max_chars=self.max_subcall_chars
                ),
                example_llm_query=(EXAMPLE_LLM_QUERY_SPLIT if split_capable else EXAMPLE_LLM_QUERY_DENSE),
            ),
        }
        begin_msg = {"role": "user", "content": "Begin. Write your first code block."}

        full_history: list = []  # all (assistant, user) messages, server-side
        evicted_count = 0  # number of leading pairs already dropped (evicted)
        forced_drop_pairs = 0  # leading pairs the SERVER rejected us for (see root_reply)
        transcript: list = []
        nudges = 0
        last_code_norm: str | None = None  # repetition-breaker state (see below)
        repeat_count = 0
        seen_output = ""  # grounding accumulator — NEVER compacted, never truncated

        def _grounded(val: str) -> bool:
            """Did this answer come from something the run actually observed?

            The task side uses word boundaries: plain containment whitelisted almost
            every short answer, since any question mentioning "12" licensed the
            ungrounded answer 12, and every multiple-choice task enumerates its
            labels in the prompt -- so that whole class of tasks had no guard at all.
            """
            return val in seen_output or _mentions(task, val)

        def build_sent() -> tuple[list, int]:
            """Construct the bounded view actually sent to the root model."""
            nonlocal evicted_count
            base = [system_msg, begin_msg]
            n_pairs = len(full_history) // 2
            pairs = [full_history[i * 2 : i * 2 + 2] for i in range(n_pairs)]
            # Pairs the SERVER made us drop (see root_reply). Applied whether or not
            # a MemoryBudget is configured -- an unbudgeted run still has a ceiling,
            # it just isn't one the operator picked.
            pairs = pairs[forced_drop_pairs:] if forced_drop_pairs else pairs

            if self.budget is None:
                fold_n = n_pairs - len(pairs)
                flat = [m for p in pairs for m in p]
                # Scratchpad notes are re-injected every turn even without a
                # budget — that visibility is the whole point of the tool.
                if self.scratchpad is not None and notes:
                    notes_msg = {
                        "role": "user",
                        "content": (
                            FOLD_SCRATCHPAD_TEMPLATE.format(n=fold_n, notes=self._notes_block(notes))
                            if fold_n
                            else SCRATCHPAD_TEMPLATE.format(notes=self._notes_block(notes))
                        ),
                    }
                    return base + [notes_msg] + flat, fold_n
                if fold_n:
                    return base + [{"role": "user", "content": FOLD_MARKER.format(n=fold_n)}] + flat, fold_n
                return base + flat, 0

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
                    msgs.append({"role": "user", "content": FOLD_MARKER.format(n=fold_n)})
                for p in kept_pairs:
                    msgs.extend(p)
                return msgs

            fold_n = n_pairs - len(kept)
            sent = assemble(kept, fold_n)
            # squeeze: drop more recent pairs until within budget
            while self.tok.count_messages(sent) > self.budget.max_context_tokens and kept:
                kept = kept[1:]
                fold_n = n_pairs - len(kept)
                sent = assemble(kept, fold_n)

            # record newly-evicted pairs (dropped for good)
            if fold_n > evicted_count:
                evicted_count = fold_n
                metrics["evictions"] += 1
            return sent, fold_n

        def root_reply() -> str:
            """Ask the root, letting the SERVER decide whether the view fits.

            Without a --max-context-tokens budget the view grows monotonically, and
            LLMClient does not retry 4xx: one overflow ended the example as
            end_reason="exception" -- the mechanism that zeroed the whole RULER-32k
            RLM run. vanilla_answer already treats the server as the authority on
            what fits; do the same here rather than requiring the operator to have
            guessed a budget in advance. Evicting the oldest turns is the same
            remedy the budget path applies, just triggered by the server.
            """
            nonlocal forced_drop_pairs
            for _ in range(12):
                sent, _ = build_sent()
                ctx_tokens = self.tok.count_messages(sent)
                try:
                    reply = self.root.chat(sent)
                except Exception as exc:  # noqa: BLE001 - re-raised unless it is an overflow
                    if not _is_context_overflow(exc) or forced_drop_pairs >= len(full_history) // 2:
                        # Nothing left to evict: the system prompt and the task alone
                        # exceed the window, which no amount of dropping can fix.
                        raise
                    forced_drop_pairs += 1
                    metrics["overflow_evictions"] += 1
                    continue
                metrics["root_prompt_tokens"] += ctx_tokens
                # Prefer the server's own count for the cost axis: TokenCounter is
                # tiktoken (a different tokenizer) or len//4 (a guess), and this
                # number is the one compared against KVPress's retained-token metric.
                served = getattr(getattr(self.root, "usage", None), "last_prompt_tokens", None)
                metrics["peak_context_tokens"] = max(metrics["peak_context_tokens"], served or ctx_tokens)
                return reply
            raise RuntimeError("root context overflow could not be resolved by eviction")

        for step in range(1, self.max_steps + 1):
            if self._deadline is not None and time.monotonic() > self._deadline:
                # Out of wall-clock. A distinct end_reason so a timeout stays
                # distinguishable from a genuine abstention when triaging.
                return RLMResult(None, step, False, transcript, "run_timeout", metrics)
            metrics["steps"] = step

            reply = root_reply()
            metrics["root_completion_tokens"] += self.tok.count(reply)
            blocks = CODE_RE.findall(reply)
            if not blocks:
                # A reply cut off at max_tokens has an opening fence and no closing
                # one. Read as "no code block" it burned a nudge, and after three
                # such replies the raw truncated text was returned as the answer.
                m_trunc = UNTERMINATED_CODE_RE.search(reply)
                if m_trunc and m_trunc.group(1).strip():
                    blocks = [m_trunc.group(1)]
                    metrics["truncated_code_blocks"] = metrics.get("truncated_code_blocks", 0) + 1

            # adaptive observation limit: never let one observation exceed the budget
            obs_limit = self.obs_limit
            if self.budget is not None:
                obs_limit = max(512, min(self.obs_limit, self.budget.max_context_tokens * CHARS_PER_TOKEN_ESTIMATE))

            # --- No code block in the reply ---
            if not blocks:
                mn = TEXT_FINAL_NONE_RE.search(reply)
                if mn:
                    # An abstention needs no grounding: "I found nothing" is exactly
                    # the claim that cannot appear in the output it is about.
                    transcript.append(
                        {
                            "step": step,
                            "reply": reply,
                            "code": None,
                            "observation": "[FINAL_NONE parsed from prose]",
                        }
                    )
                    metrics["abstain_reason"] = (mn.group(1) or "").strip()
                    return RLMResult(None, step, True, transcript, "abstained", metrics)
                m = TEXT_FINAL_RE.search(reply)
                if m:
                    val = m.group(1).strip()
                    if val and _grounded(val):
                        transcript.append(
                            {
                                "step": step,
                                "reply": reply,
                                "code": None,
                                "observation": "[FINAL parsed from prose]",
                            }
                        )
                        return RLMResult(val, step, True, transcript, "final_in_prose", metrics)
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
                        return RLMResult(None, step, False, transcript, "ungrounded_final", metrics)
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append(
                        {
                            "role": "user",
                            "content": f"REJECTED: your answer {val!r} never appeared in any actual REPL "
                            "output, so it looks like a guess. Do NOT invent answers. Write a "
                            "```python code block that finds the answer in `context` (string "
                            "search / regex / llm_query with the snippet pasted in), look at "
                            "the real output, and only then FINAL it. If you have genuinely "
                            "searched and the document does not contain the answer, say so with "
                            'FINAL_NONE("reason") instead.',
                        }
                    )
                    continue
                mv = TEXT_FINAL_VAR_RE.search(reply)
                if mv and mv.group(1) in env:
                    # Only when the variable actually exists: naming a variable that
                    # was never assigned used to answer the literal string
                    # "<missing var x>" and record it as a finished, successful
                    # prediction. A miss falls through to the no-code-block nudge.
                    val = str(env[mv.group(1)])
                    transcript.append(
                        {
                            "step": step,
                            "reply": reply,
                            "code": None,
                            "observation": "[FINAL_VAR parsed from prose]",
                        }
                    )
                    return RLMResult(val, step, True, transcript, "final_var_in_prose", metrics)
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
            obs, obs_full = self._exec(code, env, obs_limit)
            if len(blocks) > 1:
                obs += (
                    "\n[WARNING: you wrote multiple code blocks; ONLY the FIRST was "
                    "executed. Anything you wrote after it (including any 'output' you "
                    "predicted) did NOT happen.)"
                )
            transcript.append({"step": step, "reply": reply, "code": code, "observation": obs})
            # The UNtruncated output: what the model was shown is a display concern,
            # what actually printed is what the answer has to be grounded in.
            seen_output += "\n" + obs_full

            if env["_final_box"]["done"]:
                val = env["_final_box"]["value"]
                if env["_final_box"].get("abstained"):
                    metrics["abstain_reason"] = env["_final_box"].get("reason", "")
                    return RLMResult(None, step, True, transcript, "abstained", metrics)
                if val in (env.get("_rlm_final_literals") or []) and not _grounded(val):
                    env["_final_box"]["done"] = False
                    env["_final_box"]["value"] = None
                    nudges += 1
                    if nudges > 2:
                        return RLMResult(None, step, False, transcript, "ungrounded_final", metrics)
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append(
                        {
                            "role": "user",
                            "content": f"REJECTED: FINAL({val!r}) is a literal constant that never appeared "
                            "in any REPL output — it looks like a guess. Find the real answer in "
                            "`context` first (e.g. re.search / context.find / llm_query with the "
                            "snippet pasted in), print it, then call FINAL on the variable "
                            "holding it. If you have genuinely searched and the document does not "
                            'contain the answer, say so with FINAL_NONE("reason") instead.',
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
                if hasattr(self.sub, "chat_split"):
                    # Split backend: the extracted material is the compressible
                    # context, the task rides the uncompressed question side.
                    fb_question = (
                        "Answer the task using ONLY the provided material, which was "
                        "extracted from a long document by prior code. Be concise and "
                        f"factual.\n\nTask: {task}"
                    )
                    ans = self.sub.chat_split(  # type: ignore[attr-defined]
                        question=fb_question, context=material, system=SUB_SYSTEM_PROMPT
                    )
                    fb_tokens = self.tok.count(fb_question) + self.tok.count(material)
                    metrics["sub_split_calls"] = metrics.get("sub_split_calls", 0) + 1
                else:
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
                    fb_tokens = self.tok.count(fb_prompt)
                metrics["sub_calls"] += 1
                metrics["sub_call_tokens"] += fb_tokens + self.tok.count(ans)
                transcript.append(
                    {
                        "step": step,
                        "reply": "[repetition-breaker fallback]",
                        "code": None,
                        "observation": "[forced grounded answer after 3 identical code cells]",
                        # The answer was extracted BY the harness, not chosen by the
                        # model, and it skipped the grounding guard. Triage must be
                        # able to tell these apart from answers the model committed to.
                        "forced": True,
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
    client: LLMClient,
    context: str,
    task: str,
    char_limit: int = 400_000,
    answer_format: str | None = None,
    max_prompt_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    stats: Optional[dict] = None,
) -> str:
    """Baseline: stuff (possibly truncated) context directly into the prompt.

    `answer_format` is the benchmark's output contract (e.g. LOFT's
    "Final Answer: ['answer1', ...]"). It goes to BOTH arms -- the RLM gets the same
    string in ROOT_SYSTEM_PROMPT -- so neither arm is scored on an answer shape the
    other was never asked for. Without it this prompt said only "Answer concisely"
    while the RLM answered via `str(FINAL(x))`, and string-matching metrics then
    measured that style difference as if it were accuracy.

    `stats`, if given, is filled with how much context actually survived
    truncation. The caller needs this to separate "the model got it wrong" from
    "the answer was never in the prompt". Without it the vanilla arm's score is
    not comparable to a KVPress run, which always sees the full context: on a
    200k-char synthetic with a 100k limit vanilla is structurally capped, and a
    bare score column reads that cap as model accuracy.
    """
    truncated = context[:char_limit]
    fmt = f"\nFormat your answer as: {answer_format}" if answer_format else ""

    def build(body: str) -> str:
        note = "" if len(context) <= len(body) else "\n[NOTE: document truncated]"
        return f"Document:\n{body}{note}\n\nTask: {task}\nAnswer concisely.{fmt}"

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

    # The final attempt is inside the loop, not after it: a 13th call outside
    # the try raised uncaught on overflow, and record(12) then reported 12
    # retries for what was really a 13th attempt.
    last_attempt = 12
    for attempt in range(last_attempt + 1):
        try:
            out = client.chat([{"role": "user", "content": prompt}])
            record(attempt)
            return out
        except Exception as e:  # noqa: BLE001 - narrowed by the guard below
            if attempt == last_attempt or not _is_context_overflow(e) or len(truncated) <= 2000:
                record(attempt)
                raise
            truncated = truncated[: int(len(truncated) * 0.8)]
            prompt = build(truncated)
    raise AssertionError("unreachable: the loop either returns or raises")
