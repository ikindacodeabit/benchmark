# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RLM repetition breaker.

A stuck small model re-emits the same code cell every turn, producing an identical
observation each time, and never calls FINAL. Without a breaker the example burns
every one of `max_steps` and then the full `--run-timeout`, so a handful of stuck
examples can add hours to a grid. The breaker escalates: nudge on the second
identical cell, forced grounded answer on the third.
"""

import pytest

from evaluation.rlm.rlm import RLM


class FakeClient:
    """Minimal stand-in for LLMClient: records calls, replays scripted replies."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if len(self._replies) == 1:
            return self._replies[0]
        return self._replies.pop(0)


STUCK_CODE = "```python\nprint(context[:10])\n```"


def _rlm(root, sub, **kwargs):
    return RLM(
        root_client=root,
        sub_client=sub,
        max_steps=20,
        exec_timeout=None,
        run_timeout=None,
        max_sub_calls=None,
        **kwargs,
    )


def test_identical_code_three_times_triggers_forced_answer():
    root = FakeClient([STUCK_CODE])
    sub = FakeClient(["the grounded answer"])
    result = _rlm(root, sub).run("hello world, the capital is Paris", "What is the capital?")

    assert result.end_reason == "repetition_broken"
    assert result.finished is True
    assert result.answer == "the grounded answer"
    # The forced answer comes from the sub model, not the looping root.
    assert len(sub.calls) == 1


def test_breaker_stops_the_loop_early():
    """It must not run to max_steps -- that is the cost the breaker exists to avoid."""
    root = FakeClient([STUCK_CODE])
    sub = FakeClient(["answer"])
    result = _rlm(root, sub).run("some context", "a task")

    assert result.steps < 20


def test_nudge_is_sent_on_the_second_identical_cell():
    root = FakeClient([STUCK_CODE])
    sub = FakeClient(["answer"])
    _rlm(root, sub).run("some context", "a task")

    # By the third root call the conversation must contain the STOP nudge.
    last_sent = root.calls[-1]
    assert any("STOP: you just ran this EXACT same code twice" in m["content"] for m in last_sent)


def test_forced_answer_is_grounded_in_seen_output():
    """The fallback prompt must carry the REPL material, not just the task."""
    root = FakeClient([STUCK_CODE])
    sub = FakeClient(["answer"])
    _rlm(root, sub).run("MARKER_TEXT and more", "a task")

    fallback_prompt = sub.calls[0][-1]["content"]
    assert "Extracted material:" in fallback_prompt
    # STUCK_CODE prints context[:10], so the accumulated REPL output carries the
    # first ten characters of the document, not the whole marker.
    assert "MARKER_TEX" in fallback_prompt
    assert "Task: a task" in fallback_prompt


def test_varying_code_does_not_trigger_the_breaker():
    """Genuine progress must be left alone."""
    replies = [
        "```python\nprint(context[:5])\n```",
        "```python\nprint(context[5:10])\n```",
        "```python\nx = context[:5]\nFINAL(x)\n```",
    ]
    root = FakeClient(replies)
    sub = FakeClient(["unused"])
    result = _rlm(root, sub).run("hello world", "a task")

    assert result.end_reason == "final_called"
    assert sub.calls == []


def test_whitespace_only_change_still_counts_as_a_repeat():
    """Normalisation is on whitespace, so reformatting the same cell is a repeat."""
    replies = [
        "```python\nprint(context[:10])\n```",
        "```python\nprint(context[:10])\n```",
        "```python\n\nprint(context[:10])\n\n```",
        "```python\nprint(context[:10])\n```",
    ]
    root = FakeClient(replies)
    sub = FakeClient(["answer"])
    result = _rlm(root, sub).run("hello world", "a task")

    assert result.end_reason == "repetition_broken"


@pytest.mark.parametrize("end_reason", ["repetition_broken"])
def test_end_reason_is_distinct_from_max_steps(end_reason):
    """Distinguishable in metrics, so stuck examples can be counted separately."""
    root = FakeClient([STUCK_CODE])
    sub = FakeClient(["answer"])
    result = _rlm(root, sub).run("ctx", "task")
    assert result.end_reason == end_reason != "max_steps"
