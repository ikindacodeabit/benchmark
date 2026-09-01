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


# --- unproductive-step breaker ------------------------------------------------
# Every cell below is textually DIFFERENT, so the repetition breaker cannot fire
# on any of them -- which is the point. This is the shape that got past it on the
# LOFT-1m smoke: comment-only cells rambling toward a remembered answer, printing
# nothing, 11 of 13 steps, until run_timeout killed the example 32 minutes in.
RAMBLING_CELLS = [
    "```python\n# the dwarf is probably Brienne\n```",
    "```python\n# no wait, it is Tormund\n```",
    "```python\n# actually, on reflection, Tyrion\n```",
    "```python\n# let me reconsider once more\n```",
]


def test_three_cells_printing_nothing_trigger_a_forced_answer():
    root = FakeClient(list(RAMBLING_CELLS))
    sub = FakeClient(["the grounded answer"])
    result = _rlm(root, sub).run("hello world, the capital is Paris", "What is the capital?")

    assert result.end_reason == "unproductive_broken"
    assert result.finished is True
    assert result.answer == "the grounded answer"


def test_it_breaks_sooner_than_the_repetition_breaker_would():
    """The reason this breaker exists. Every cell differs, so the repetition
    breaker cannot fire until the model happens to repeat one -- by which point
    the quiet steps have already been paid for. On LOFT-1m that was 13 steps and
    32 minutes for an answer of 0."""
    assert len(set(RAMBLING_CELLS)) == len(RAMBLING_CELLS)

    with_guard = _rlm(FakeClient(list(RAMBLING_CELLS)), FakeClient(["a"])).run("ctx", "task")
    without = _rlm(
        FakeClient(list(RAMBLING_CELLS)), FakeClient(["a"]), max_unproductive_steps=0
    ).run("ctx", "task")

    assert with_guard.end_reason == "unproductive_broken"
    assert with_guard.steps < without.steps


def test_nudge_is_sent_one_step_before_the_break():
    root = FakeClient(list(RAMBLING_CELLS))
    sub = FakeClient(["answer"])
    _rlm(root, sub).run("some context", "a task")

    sent = "".join(m["content"] for call in root.calls for m in call if m["role"] == "user")
    assert "printed NOTHING" in sent


def test_a_cell_that_prints_resets_the_counter():
    """Genuine progress between quiet cells must not accumulate toward the break."""
    replies = [
        "```python\n# thinking\n```",
        "```python\n# still thinking\n```",
        "```python\nprint(context[:5])\n```",
        "```python\n# thinking again\n```",
        "```python\n# and again\n```",
        "```python\nx = context[:5]\nFINAL(x)\n```",
    ]
    root = FakeClient(replies)
    sub = FakeClient(["unused"])
    result = _rlm(root, sub).run("hello world", "a task")

    assert result.end_reason == "final_called"
    assert sub.calls == []


def test_zero_disables_the_breaker():
    replies = list(RAMBLING_CELLS) + ["```python\nx = context[:5]\nFINAL(x)\n```"]
    root = FakeClient(replies)
    sub = FakeClient(["unused"])
    result = _rlm(root, sub, max_unproductive_steps=0).run("hello world", "a task")

    assert result.end_reason == "final_called"
    assert sub.calls == []


# --- repeated-exception breaker -----------------------------------------------
# The third route to the same dead end, and the one that cost the most: on LOFT-1m
# a single example threw on 34 of its 38 steps and burned 73% of the whole run's
# tokens before the repetition breaker happened to catch it at step 37.
RAISING_CELLS = [
    "```python\nprint(context[0:2y])\n```",  # SyntaxError: invalid decimal literal
    "```python\nprint(undefined_name_one)\n```",  # NameError
    "```python\nprint(1/0)\n```",  # ZeroDivisionError
    "```python\nprint(undefined_name_two)\n```",
]


def test_three_raising_cells_trigger_a_forced_answer():
    root = FakeClient(list(RAISING_CELLS))
    sub = FakeClient(["the grounded answer"])
    result = _rlm(root, sub).run("hello world, the capital is Paris", "What is the capital?")

    assert result.end_reason == "error_loop_broken"
    assert result.finished is True
    assert result.answer == "the grounded answer"


def test_an_exception_does_not_count_as_a_quiet_cell():
    """Guards the reason this breaker is separate: an exception IS output, so the
    unproductive counter resets on every one and never reaches its threshold."""
    root = FakeClient(list(RAISING_CELLS))
    sub = FakeClient(["answer"])
    result = _rlm(root, sub, max_error_steps=0).run("ctx", "task")

    assert result.end_reason != "unproductive_broken"


def test_it_breaks_sooner_than_the_repetition_breaker_would_on_broken_code():
    assert len(set(RAISING_CELLS)) == len(RAISING_CELLS)

    with_guard = _rlm(FakeClient(list(RAISING_CELLS)), FakeClient(["a"])).run("ctx", "task")
    without = _rlm(FakeClient(list(RAISING_CELLS)), FakeClient(["a"]), max_error_steps=0).run(
        "ctx", "task"
    )

    assert with_guard.end_reason == "error_loop_broken"
    assert with_guard.steps < without.steps


def test_a_cell_that_runs_resets_the_error_counter():
    replies = [
        "```python\nprint(undefined_one)\n```",
        "```python\nprint(undefined_two)\n```",
        "```python\nprint(context[:5])\n```",
        "```python\nprint(undefined_three)\n```",
        "```python\nprint(undefined_four)\n```",
        "```python\nx = context[:5]\nFINAL(x)\n```",
    ]
    root = FakeClient(replies)
    sub = FakeClient(["unused"])
    result = _rlm(root, sub).run("hello world", "a task")

    assert result.end_reason == "final_called"
    assert sub.calls == []


def test_nudge_names_the_terminal_calls():
    root = FakeClient(list(RAISING_CELLS))
    sub = FakeClient(["answer"])
    _rlm(root, sub).run("some context", "a task")

    sent = "".join(m["content"] for call in root.calls for m in call if m["role"] == "user")
    assert "all raised an exception" in sent


# --- FINAL spelling aliases ---------------------------------------------------
@pytest.mark.parametrize("name", ["FINAL", "Final", "final"])
def test_final_accepts_case_variants(name):
    """A mis-cased FINAL used to raise NameError and burn a step."""
    root = FakeClient([f"```python\nx = context[:5]\n{name}(x)\n```"])
    sub = FakeClient(["unused"])
    result = _rlm(root, sub).run("hello world", "a task")

    assert result.end_reason == "final_called"
    assert result.answer == "hello"


@pytest.mark.parametrize("name", ["FINAL_NONE", "Final_None", "final_none"])
def test_final_none_accepts_case_variants(name):
    root = FakeClient([f"```python\n{name}('not here')\n```"])
    sub = FakeClient(["unused"])
    result = _rlm(root, sub, min_sub_calls_before_abstain=0).run("hello world", "a task")

    assert result.end_reason == "abstained"


@pytest.mark.parametrize("name", ["FINAL", "Final", "final"])
def test_the_grounding_guard_still_covers_every_spelling(name):
    """The alias must not become a hole in the anti-guessing check: a literal the
    model never saw has to be rejected however FINAL is spelled."""
    root = FakeClient([f"```python\n{name}('Tyrion Lannister')\n```"])
    sub = FakeClient(["unused"])
    result = _rlm(root, sub).run("a document mentioning nobody", "who?")

    assert result.answer != "Tyrion Lannister"


def test_end_reason_separates_going_quiet_from_getting_stuck():
    """Both are forced answers, but they call for different fixes, so a campaign
    must be able to count them apart."""
    quiet = _rlm(FakeClient(list(RAMBLING_CELLS)), FakeClient(["a"])).run("ctx", "task")
    stuck = _rlm(FakeClient([STUCK_CODE]), FakeClient(["a"])).run("ctx", "task")

    assert quiet.end_reason == "unproductive_broken"
    assert stuck.end_reason == "repetition_broken"
