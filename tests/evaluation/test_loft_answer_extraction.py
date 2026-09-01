# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for LOFT answer extraction across the vanilla and RLM arms.

The bug these guard against: `calculate_metrics.extract_prediction` recovers a
prediction only from a bracketed list or a cued line. The RLM arm answers with a
bare `str(FINAL(x))`, so extraction returned [] and every RLM row scored 0.0
regardless of correctness, while cued baseline prose scored normally. That is an
arm asymmetry large enough to invert the conclusion of a vanilla-vs-RLM run.
"""

import pandas as pd
import pytest

from evaluation.benchmarks.loft.answer_extraction import extract_answers, strip_reasoning
from evaluation.benchmarks.loft.calculate_metrics import calculate_metrics


def _frame(predicted_answer, answers, task="nq_128k", answer_prefix="Final Answer: "):
    return pd.DataFrame(
        [
            {
                "predicted_answer": predicted_answer,
                "answers": answers,
                "task": task,
                "answer_prefix": answer_prefix,
            }
        ]
    )


# --------------------------------------------------------------------------
# The core asymmetry: both arms must score the same correct answer.
# --------------------------------------------------------------------------


def test_rlm_bare_answer_scores_nonzero():
    """A bare `str(FINAL(x))` with no cue and no brackets must not score 0."""
    metrics = calculate_metrics(_frame("Barack Obama", ["Barack Obama"]))
    assert metrics["em"] == pytest.approx(1.0)
    assert metrics["subspan_em"] == pytest.approx(1.0)


def test_cued_and_bare_answers_score_identically():
    cued = calculate_metrics(_frame("Final Answer: Barack Obama", ["Barack Obama"]))
    bare = calculate_metrics(_frame("Barack Obama", ["Barack Obama"]))
    assert cued["em"] == bare["em"]
    assert cued["subspan_em"] == bare["subspan_em"]
    assert cued["f1"] == bare["f1"]


def test_bare_wrong_answer_still_scores_zero():
    """The fallback must not be so lenient that everything passes."""
    metrics = calculate_metrics(_frame("Abraham Lincoln", ["Barack Obama"]))
    assert metrics["em"] == pytest.approx(0.0)
    assert metrics["subspan_em"] == pytest.approx(0.0)


def test_empty_prediction_scores_zero():
    metrics = calculate_metrics(_frame("", ["Barack Obama"]))
    assert metrics["em"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Multi-value tasks (qampari/quest) take the list path.
# --------------------------------------------------------------------------


def test_multi_value_bare_list_repr_scores():
    """RLM's `str(FINAL([...]))` renders a Python list repr with no cue."""
    metrics = calculate_metrics(
        _frame("['France', 'Spain', 'Italy']", ["France", "Spain", "Italy"], task="qampari_128k")
    )
    assert metrics["coverage"] == pytest.approx(1.0)
    assert "f1" not in metrics


def test_multi_value_comma_prose_is_not_split_by_the_primary_extractor():
    """Documents a LOFT-faithful behaviour that is easy to mistake for a bug.

    LOFT asks for a bracketed list. `extract_prediction` finds the cue, returns the
    whole tail as ONE answer, and multi-value scoring then compares that single
    string against N golds -> coverage 0. Because the primary extractor succeeded,
    our fallback never runs.

    This is deliberately left alone: LOFT's official evaluation penalises the same
    non-compliant output, and "fixing" it here would silently diverge from the
    published qampari/quest numbers. It is a real model-behaviour difference, not a
    harness artifact -- but it does mean an arm that emits a proper list repr scores
    higher on qampari/quest than one that answers in prose. Interpret those two
    datasets with that in mind.
    """
    metrics = calculate_metrics(
        _frame("Final Answer: France, Spain, Italy", ["France", "Spain", "Italy"], task="quest_128k")
    )
    assert metrics["coverage"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Behaviour that must NOT change: replies the primary extractor already handles.
# --------------------------------------------------------------------------


def test_primary_extractor_still_wins_on_bracketed_output():
    """The fallback runs only when the primary extractor returns nothing."""
    metrics = calculate_metrics(_frame("Final Answer: [Paris]", ["Paris"], task="nq_128k"))
    assert metrics["em"] == pytest.approx(1.0)


def test_fallback_does_not_run_list_parsing_for_single_value_tasks():
    """A stray bracket must not hijack a single-value answer in the fallback.

    (The primary extractor does grab `[1]` from `Final Answer: Paris [1]` -- that is
    LOFT-official behaviour, since LOFT asks for the answer *as* a bracketed list.
    We leave it alone; this test pins the fallback's own behaviour, which is only
    reached when the primary finds nothing.)
    """
    assert extract_answers("Paris [1]", answer_prefix="Final Answer: ") == ["Paris [1]"]
    assert extract_answers("Paris [1]", expect_list=True) == ["1"]


# --------------------------------------------------------------------------
# extract_answers unit behaviour.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<think>hmm</think>Paris", "Paris"),
        ("reasoning here</think>Paris", "Paris"),
        ("Paris<think>unclosed tail", "Paris"),
        ("", ""),
        (None, ""),
    ],
)
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_cue_mentioned_with_no_content_falls_back_to_earlier_occurrence():
    text = "Final Answer: Paris\nI could not determine the Final Answer:"
    assert extract_answers(text, answer_prefix="Final Answer: ") == ["Paris"]


def test_nested_list_is_flattened():
    assert extract_answers("[['a', 'b'], ['c']]", expect_list=True) == ["a", "b", "c"]


def test_bracket_inside_string_literal_does_not_break_parse():
    assert extract_answers('["a [ b", "c"]', expect_list=True) == ["a [ b", "c"]


def test_enumerated_prose_becomes_a_list():
    raw = "Final Answer:\n1. France\n2. Spain\n3. Italy"
    assert extract_answers(raw, answer_prefix="Final Answer: ", expect_list=True) == [
        "France",
        "Spain",
        "Italy",
    ]


def test_empty_prediction_returns_empty_list():
    assert extract_answers(None) == []
    assert extract_answers("   ") == []


# --- empty predictions must never score on the multi-value subsets -------------
# qampari and quest score through compute_multi_value_subspan_em, whose match test
# runs BOTH directions (`gold in pred or pred in gold`). The empty string is a
# substring of every gold answer, so it matched everything: "Final Answer: []"
# extracts to ["[]"], normalize_answers strips that to [""], and a single-item gold
# list scored a full 1.0 for answering nothing. On the LOFT-1m quest run that was
# the baseline's ENTIRE score -- 23 of 110 rows, every one an empty prediction,
# reported as 0.209 against a true 0.000, which inverted the arm comparison.


@pytest.mark.parametrize("task", ["qampari_1m", "quest_1m"])
def test_empty_list_answer_scores_zero_against_one_gold(task):
    """The exact shape that inflated the LOFT-1m baseline."""
    m = calculate_metrics(_frame("Final Answer: []", ["Spartacus (film)"], task=task))

    assert m["subspan_em"] == 0.0
    assert m["em"] == 0.0


@pytest.mark.parametrize("task", ["qampari_1m", "quest_1m"])
def test_empty_list_answer_scores_zero_against_several_golds(task):
    m = calculate_metrics(_frame("Final Answer: []", ["Laal Paree", "Roja (film)"], task=task))

    assert m["subspan_em"] == 0.0


def test_a_genuine_multi_value_match_still_scores():
    """Guards against fixing the hole by breaking real matches."""
    m = calculate_metrics(
        _frame("Final Answer: ['Spartacus (film)']", ["Spartacus (film)"], task="quest_1m")
    )

    assert m["subspan_em"] == 1.0


def test_a_genuine_subspan_still_scores():
    """The bidirectional test is the point of subspan_em -- a prediction that
    contains the gold, or is contained by it, must still count."""
    m = calculate_metrics(_frame("Final Answer: ['Spartacus']", ["Spartacus (film)"], task="quest_1m"))

    assert m["subspan_em"] == 1.0


def test_the_single_value_path_was_never_affected():
    """nq/hotpotqa/musique go through compute_subspan_em, which tests one
    direction only (`gold in pred`), so an empty prediction never scored there.
    Pinned so the two paths cannot silently converge on the broken behaviour."""
    m = calculate_metrics(_frame("Final Answer: []", ["Tyrion Lannister"], task="nq_1m"))

    assert m["subspan_em"] == 0.0
