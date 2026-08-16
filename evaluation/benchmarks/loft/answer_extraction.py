# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fallback answer parsing for LOFT, applied identically to every arm.

WHY THIS EXISTS. ``calculate_metrics.extract_prediction`` recovers a prediction
only when the reply contains a bracketed Python list on one line, or a line
carrying the answer cue. That covers a prompted baseline, which is asked for
``Final Answer: ...`` and complies. It does not cover the RLM arm, whose native
prediction is ``str(FINAL(x))`` -- a bare answer with no cue and no brackets.
Extraction returned ``[]`` there, and ``calculate_metrics`` scores an empty
extraction as 0.0, so **every RLM row scored zero regardless of correctness**
while baseline rows scored normally.

This module is the fallback used only when the primary extractor finds nothing,
so a reply that already parses is untouched and previously published numbers move
only where extraction was failing outright.

SCOPE -- read before reusing elsewhere. LOFT's official evaluation
(google-deepmind/loft) parses the output into an answer list BEFORE scoring, so
parsing IS part of that benchmark's metric definition. LongBench and RULER
instead define their metrics over the RAW generation; running predictions through
an extractor there would diverge from the published numbers. Keep this LOFT-only.

Ported from the standalone RLM pipeline (``benchmarks/answer_extraction.py`` in
ikindacodeabit/rlm_research), where the arm-asymmetry reasoning below was worked
out against real transcripts.
"""

from __future__ import annotations

import ast
import re
from typing import Iterator, List, Optional

# Qwen3 and similar models emit a visible reasoning block. It is normally stripped
# server-side (vLLM served with --reasoning-parser, and the client returns
# message.content rather than reasoning_content), so this is a defensive guard for
# hosted-API paths and for any run that omits the flag. Qwen3-*-Instruct-2507 is a
# non-thinking model and produces none of these, making this a no-op there.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_THINK_LEADING_CLOSE_RE = re.compile(r"\A.*?</think>", re.DOTALL | re.IGNORECASE)

_LIST_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def strip_reasoning(text: Optional[str]) -> str:
    """Remove ``<think>...</think>`` reasoning from a raw completion.

    Handles three shapes, all of which occur in practice:
      * a well-formed block;
      * a stray leading ``</think>`` (the server consumed the opening tag);
      * an UNCLOSED ``<think>`` (generation hit max_tokens mid-thought) -- everything
        from the tag onward is reasoning, so it is dropped rather than scored.
    """
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub(" ", text)
    while "</think>" in out:
        out = _THINK_LEADING_CLOSE_RE.sub(" ", out, count=1)
    out = _THINK_OPEN_RE.sub(" ", out)
    return out.strip()


def _iter_balanced_lists(text: str) -> Iterator[str]:
    """Yield substrings of ``text`` that look like balanced ``[...]`` spans.

    Quote-aware: brackets inside string literals are ignored, so ``["a [ b", "c"]``
    parses instead of confusing the depth counter. Candidates are yielded
    OUTERMOST-first (left to right by start), so a nested list is returned whole
    rather than having its last inner list win.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth, quote, esc, j = 0, "", False, i
        while j < n:
            ch = text[j]
            if quote:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
            j += 1
        i += 1


def _parse_list(text: str) -> Optional[List[str]]:
    """Return the first parseable balanced Python list in ``text``, else None."""
    for candidate in _iter_balanced_lists(text):
        try:
            val = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(val, (list, tuple)):
            # Flatten one level: a model that emits [["a","b"],["c","d"]] means four
            # answers, not two stringified sublists. LOFT gold answers are always
            # flat strings, so a nested list is malformed output to be recovered.
            items: List[str] = []
            for v in val:
                for x in v if isinstance(v, (list, tuple)) else [v]:
                    s = str(x).strip()
                    if s:
                        items.append(s)
            if items:
                return items
    return None


def _after_prefix(text: str, answer_prefix: str) -> Optional[str]:
    """Text following the LAST occurrence of ``answer_prefix`` that has content."""
    key = answer_prefix.strip().rstrip(":").strip()
    if not key:
        return None
    low_text, low_key = text.lower(), key.lower()
    idx = low_text.rfind(low_key)
    while idx >= 0:
        after = text[idx + len(low_key) :].lstrip(" :\t\n")
        if after.strip():
            return after
        # The model merely MENTIONED the cue with nothing after it ("I cannot find
        # the Final Answer"). Fall back to an earlier occurrence rather than
        # abandoning the prefix logic and scoring the whole reply.
        idx = low_text.rfind(low_key, 0, idx)
    return None


def extract_answers(
    prediction: Optional[str],
    *,
    answer_prefix: Optional[str] = None,
    expect_list: bool = False,
) -> List[str]:
    """Parse a raw prediction from EITHER arm into LOFT's answer list.

    Resolution order:
      1. text after ``answer_prefix`` (the cue LOFT asks both arms to emit);
      2. within that, a bracketed Python list when ``expect_list`` (multi-value);
      3. otherwise the first line of the cued text.

    The list parse is gated on ``expect_list`` and runs only on cued text. Running
    it unconditionally meant any stray bracket hijacked the answer -- a citation
    marker turned ``"Final Answer: Paris [1]"`` into ``["1"]``, scoring a correct
    answer 0.0.

    KNOWN LENIENCY, deliberately kept: when no cue is present the WHOLE reply is
    treated as the answer, so a model that merely quotes retrieved text containing
    the gold can score under subspan-EM containment. Rejecting uncued replies was
    tried and is worse: the RLM's native answer is ``str(FINAL(x))``, which carries
    no cue, so strictness zeroed correct RLM answers while cued baseline prose
    passed -- a larger asymmetry than the one it removed.

    Returns ``[]`` for an empty/None prediction; the caller scores that as 0.0.
    """
    text = strip_reasoning(prediction)
    if not text:
        return []

    cued = _after_prefix(text, answer_prefix) if answer_prefix else None
    body = cued if cued is not None else text

    if expect_list:
        items = _parse_list(body)
        if items is not None:
            return items

    if cued is not None and not expect_list:
        body = body.split("\n", 1)[0]
    body = body.strip()
    if not body:
        return []

    if expect_list:
        # An enumerated or comma-joined answer is the natural PROSE form of a list;
        # without this, only the RLM's str(FINAL([...])) would ever parse and the
        # extraction step would itself become an arm asymmetry.
        lines = [_LIST_BULLET_RE.sub("", ln).strip() for ln in body.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) > 1:
            return lines
        if len(lines) == 1 and "," in lines[0]:
            parts = [p.strip() for p in lines[0].split(",") if p.strip()]
            if len(parts) > 1:
                return parts
        body = lines[0] if lines else body

    return [body] if body else []
