# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `search` REPL primitive and the coverage attribution around it."""

import pytest

from evaluation.rlm import retrieval
from evaluation.rlm.rlm import HIT_SEPARATOR, RLM, SUB_FIT_FAILURE_PREFIX

DOCUMENT = (
    ("filler text about nothing in particular " * 40)
    + "ZEBRAQUUX the Riverside round was won by Bignotti Cotter Racing in 1983. "
    + ("more filler text about other matters " * 40)
    + "PLUGHWOMBAT the Amity summer attacks were reported by the county sheriff. "
    + ("still more unrelated padding here " * 40)
)


class FakeClient:
    """A plain chat backend -- no `chat_split`, like the real LLMClient."""

    def __init__(self, reply="sub answer"):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


class FakeSplitClient(FakeClient):
    """A split-capable backend, like KVzipSubClient."""

    def __init__(self, reply="sub answer"):
        super().__init__(reply)
        self.split_calls = []

    def chat_split(self, question, context, system=None, **kw):
        self.split_calls.append((question, context))
        return self.reply


@pytest.fixture(autouse=True)
def _clean_cache():
    retrieval.reset_cache()
    yield
    retrieval.reset_cache()


def build(sub=None, **kw):
    """An RLM plus the pieces _make_env needs, without running the loop."""
    root = FakeClient()
    rlm = RLM(root_client=root, sub_client=sub or FakeClient(), search_k=kw.pop("search_k", 3), **kw)
    metrics = {
        "steps": 0,
        "sub_calls": 0,
        "sub_call_tokens": 0,
        "sub_cache_hits": 0,
        "sub_fit_failures": 0,
        "sub_slice_unlocatable_calls": 0,
        "document_coverage_fraction": 0.0,
        "evictions": 0,
        "overflow_evictions": 0,
    }
    env = rlm._make_env(DOCUMENT, metrics, {}, [])
    return rlm, env, metrics


class TestSearchBinding:
    def test_search_is_absent_when_retrieval_is_off(self):
        _, env, _ = build(search_k=0)
        assert "search" not in env

    def test_search_is_bound_when_retrieval_is_on(self):
        _, env, _ = build(search_k=3)
        assert callable(env["search"])

    def test_hits_are_verbatim_slices_of_the_document(self):
        _, env, _ = build()
        for hit in env["search"]("ZEBRAQUUX Riverside Bignotti"):
            assert hit.text == DOCUMENT[hit.start : hit.end]

    def test_the_model_cannot_raise_k_above_the_configured_axis(self):
        """k is the grid's swept axis. The root copies the worked example verbatim,
        so an example showing `k=100` must not be able to override the cell."""
        _, env, metrics = build(search_k=2)
        assert len(env["search"]("filler text", k=100)) == 2
        assert metrics["search_k_clamped_calls"] == 1

    def test_the_model_may_ask_for_fewer(self):
        _, env, metrics = build(search_k=5)
        assert len(env["search"]("filler text", k=1)) == 1
        assert metrics.get("search_k_clamped_calls", 0) == 0

    def test_search_calls_are_counted_and_capped(self):
        _, env, metrics = build(search_k=2, max_search_calls=2)
        env["search"]("filler")
        env["search"]("filler")
        assert env["search"]("filler") == []  # the guard, returning no hits
        assert metrics["search_calls"] == 2


class TestCoverageAttribution:
    def test_a_hit_list_payload_is_attributed_to_its_spans(self):
        _, env, metrics = build(sub=FakeSplitClient())
        hits = env["search"]("ZEBRAQUUX Riverside")
        env["llm_query"]("who won?", hits)
        expected = sum(e - s for s, e in _merged(hits)) / len(DOCUMENT)
        assert metrics["document_coverage_fraction"] == pytest.approx(expected)
        assert metrics["sub_slice_unlocatable_calls"] == 0

    def test_the_dense_fold_path_attributes_coverage_too(self):
        """The regression guard for the structural zero: LLMClient has no
        chat_split, so the slice is folded into the question and `chunk` becomes
        None. Deriving the span after that fold made document_coverage_fraction
        0.0 on every --sub-backend http run ever recorded."""
        _, env, metrics = build(sub=FakeClient())
        hits = env["search"]("ZEBRAQUUX Riverside")
        env["llm_query"]("who won?", hits)
        assert metrics["document_coverage_fraction"] > 0

    def test_a_plain_string_slice_still_attributes_on_the_fold_path(self):
        _, env, metrics = build(sub=FakeClient())
        env["llm_query"]("who won?", DOCUMENT[1000:3000])
        assert metrics["document_coverage_fraction"] == pytest.approx(2000 / len(DOCUMENT))

    def test_overlapping_hits_are_merged_rather_than_double_counted(self):
        _, env, metrics = build(sub=FakeSplitClient(), search_k=5)
        hits = env["search"]("filler text about nothing")
        env["llm_query"]("what?", hits)
        naive = sum(h.end - h.start for h in hits) / len(DOCUMENT)
        assert metrics["document_coverage_fraction"] <= naive + 1e-9

    def test_coverage_is_not_recorded_when_the_sub_model_refuses(self):
        sub = FakeSplitClient(reply=f"{SUB_FIT_FAILURE_PREFIX} slice too large")
        _, env, metrics = build(sub=sub)
        hits = env["search"]("ZEBRAQUUX Riverside")
        env["llm_query"]("who won?", hits)
        assert metrics["document_coverage_fraction"] == 0.0
        assert metrics["sub_fit_failures"] == 1

    def test_truncation_clips_the_attributed_span(self):
        _, env, metrics = build(sub=FakeSplitClient(), max_subcall_chars=900)
        hits = env["search"]("ZEBRAQUUX Riverside")
        env["llm_query"]("who won?", hits)
        covered = metrics["document_coverage_fraction"] * len(DOCUMENT)
        assert 0 < covered <= 900
        assert metrics["sub_payload_truncated_calls"] == 1

    def test_an_empty_hit_list_sends_the_question_alone(self):
        sub = FakeSplitClient()
        _, env, _ = build(sub=sub)
        env["llm_query"]("who won?", [])
        assert sub.calls and "[]" not in sub.calls[0][-1]["content"]

    def test_a_hand_joined_payload_records_no_coverage(self):
        """The residual hole, documented rather than hidden: a payload the MODEL
        assembles cannot be located in the document."""
        _, env, metrics = build(sub=FakeSplitClient())
        hits = env["search"]("ZEBRAQUUX Riverside")
        env["llm_query"]("who won?", "\n".join(h.text for h in hits))
        assert metrics["document_coverage_fraction"] == 0.0

    def test_hits_read_counts_what_was_handed_to_the_sub_model(self):
        _, env, metrics = build(sub=FakeSplitClient(), search_k=3)
        hits = env["search"]("filler text")
        env["llm_query"]("what?", hits)
        assert metrics["search_hits_read"] == len(hits)

    def test_retrieved_spans_are_recorded_for_the_gold_ceiling(self):
        _, env, metrics = build()
        env["search"]("ZEBRAQUUX Riverside")
        assert metrics["search_retrieved_spans"]
        assert all(0 <= s < e <= len(DOCUMENT) for s, e in metrics["search_retrieved_spans"])


class TestFloorInteraction:
    def test_the_floor_widens_hit_spans_and_they_stay_verbatim(self):
        # One hit is `search_window` wide (2000); the floor must grow it past that
        # without saturating this deliberately small document.
        rlm, env, metrics = build(
            sub=FakeSplitClient(), search_k=1, min_subcall_chars=3000, max_subcall_chars=60000
        )
        hits = env["search"]("ZEBRAQUUX Riverside")
        env["llm_query"]("who won?", hits)
        covered = metrics["document_coverage_fraction"] * len(DOCUMENT)
        assert 3000 <= covered < len(DOCUMENT)
        assert metrics["sub_slice_unlocatable_calls"] == 0

    def test_a_hit_payload_is_never_counted_unlocatable(self):
        _, env, metrics = build(
            sub=FakeSplitClient(), search_k=1, min_subcall_chars=3000, max_subcall_chars=60000
        )
        env["llm_query"]("who won?", env["search"]("ZEBRAQUUX"))
        assert metrics["sub_slice_unlocatable_calls"] == 0


class TestPromptWiring:
    def test_the_search_prompt_replaces_the_find_demonstration(self):
        rlm, _, _ = build(search_k=4)
        slots = rlm._read_instructions(False)
        assert "hits = search(" in slots["worked_session"]
        assert 'context.find("invoice total")' not in slots["worked_session"]
        assert "4 best-matching windows" in slots["search_terms"]

    def test_retrieval_off_keeps_the_find_demonstration(self):
        rlm, _, _ = build(search_k=0)
        slots = rlm._read_instructions(False)
        assert 'context.find("invoice total")' in slots["worked_session"]
        assert "search(" not in slots["search_terms"]

    def test_the_search_help_does_not_promise_cheap_large_slices_on_http(self):
        """That claim is only true when the sub backend compresses its context."""
        rlm, _, _ = build(search_k=4)
        assert "costs barely more" not in rlm._read_instructions(False)["llm_query_help"]


def _merged(hits):
    spans = sorted((h.start, h.end) for h in hits)
    out = []
    for start, end in spans:
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def test_the_separator_is_what_clipping_charges_for():
    assert HIT_SEPARATOR
