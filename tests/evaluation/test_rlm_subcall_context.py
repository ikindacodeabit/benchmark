# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the split (question, context) llm_query path used by the kvpress
sub-backend.

A split-capable sub client receives the context slice as a separate argument so
the press can compress it apart from the question. These tests pin three things:
the dispatch contract, the fallback behavior on plain clients, and the DENSE
prompt rendering — which is pinned so that a wording change is always a
deliberate act, since it re-baselines every arm that reads it.
"""

import time

from evaluation.rlm.rlm import (
    EXAMPLE_LLM_QUERY_DENSE,
    EXAMPLE_LLM_QUERY_SPLIT,
    LLM_QUERY_HELP_DENSE,
    READ_STRATEGY_DEFAULT,
    RLM,
    SUB_SYSTEM_PROMPT,
)


class FakeClient:
    """Plain LLMClient stand-in: chat only, records calls."""

    def __init__(self, replies=("sub answer",)):
        self._replies = list(replies)
        self.chat_calls = []

    def _pop(self):
        if len(self._replies) == 1:
            return self._replies[0]
        return self._replies.pop(0)

    def chat(self, messages, **kwargs):
        self.chat_calls.append(messages)
        return self._pop()


class FakeSplitClient(FakeClient):
    """kvpress-backend stand-in: additionally exposes chat_split."""

    def __init__(self, replies=("sub answer",)):
        super().__init__(replies)
        self.split_calls = []

    def chat_split(self, question, context, system=None, **kwargs):
        self.split_calls.append({"question": question, "context": context, "system": system})
        return self._pop()


def _rlm(sub, **kwargs):
    return RLM(
        root_client=FakeClient(),
        sub_client=sub,
        exec_timeout=None,
        run_timeout=None,
        max_sub_calls=None,
        **kwargs,
    )


def _metrics():
    return {"sub_calls": 0, "sub_call_tokens": 0, "sub_cache_hits": 0}


def _llm_query(rlm, metrics=None, cache=None, document="the full document"):
    metrics = _metrics() if metrics is None else metrics
    cache = {} if cache is None else cache
    env = rlm._make_env(document, metrics, cache, [])
    return env["llm_query"], metrics, cache


def test_two_arg_call_dispatches_to_chat_split_with_the_system_prompt():
    sub = FakeSplitClient()
    llm_query, metrics, _ = _llm_query(_rlm(sub))

    assert llm_query("What is X?", "a slice of the document") == "sub answer"
    assert sub.chat_calls == []
    (call,) = sub.split_calls
    assert call["question"] == "What is X?"
    assert call["context"] == "a slice of the document"
    assert call["system"] == SUB_SYSTEM_PROMPT
    assert metrics["sub_calls"] == 1
    assert metrics["sub_split_calls"] == 1


def test_one_arg_call_on_a_split_client_still_uses_plain_chat():
    sub = FakeSplitClient()
    llm_query, metrics, _ = _llm_query(_rlm(sub))

    llm_query("just a question")
    assert sub.split_calls == []
    (messages,) = sub.chat_calls
    assert messages[0] == {"role": "system", "content": SUB_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "just a question"}
    assert "sub_split_calls" not in metrics


def test_context_is_truncated_and_the_notice_rides_the_question_side():
    sub = FakeSplitClient()
    llm_query, _, _ = _llm_query(_rlm(sub, max_subcall_chars=100))

    llm_query("Q", "x" * 500)
    (call,) = sub.split_calls
    # The slice gets whatever the question leaves of the ONE budget: 100 - len("Q").
    assert len(call["context"]) == 99
    assert "[NOTE: your prompt was truncated at 100 chars" in call["question"]
    assert "[NOTE:" not in call["context"]


def test_the_two_sides_share_one_budget_rather_than_getting_one_each():
    """Capping each side at max_subcall_chars let a split call carry ~2x the size
    the prompt advertises -- which is what the sub model's KV fit check then
    rejects, with a string the root is free to ignore."""
    sub = FakeSplitClient()
    llm_query, _, _ = _llm_query(_rlm(sub, max_subcall_chars=1000))

    llm_query("q" * 5000, "x" * 5000)
    (call,) = sub.split_calls
    question_without_notice = call["question"].split("\n[NOTE:")[0]
    assert len(question_without_notice) + len(call["context"]) <= 1000
    # The question is an instruction, not the payload: it gets a quarter, the
    # slice takes the rest, so a long question cannot squeeze the slice to nothing.
    assert len(question_without_notice) == 250
    assert len(call["context"]) == 750


def test_cache_distinguishes_the_same_question_over_different_slices():
    sub = FakeSplitClient(replies=["first", "second"])
    llm_query, metrics, _ = _llm_query(_rlm(sub))

    assert llm_query("Q", "slice A") == "first"
    assert llm_query("Q", "slice B") == "second"
    # Repeat of the first pair must come from the cache, not a third call.
    assert llm_query("Q", "slice A") == "first"
    assert len(sub.split_calls) == 2
    assert metrics["sub_cache_hits"] == 1


def test_two_arg_call_on_a_plain_client_folds_context_into_the_prompt():
    sub = FakeClient()
    llm_query, metrics, _ = _llm_query(_rlm(sub))

    assert llm_query("What is X?", "a slice") == "sub answer"
    (messages,) = sub.chat_calls
    assert messages[1]["content"] == "What is X?\n\na slice"
    assert metrics["sub_calls"] == 1


def test_deadline_check_is_opt_in_and_blocks_the_call_when_expired():
    sub = FakeSplitClient()
    rlm = _rlm(sub, subcall_deadline_check=True)
    rlm._deadline = time.monotonic() - 1
    llm_query, _, _ = _llm_query(rlm)

    ans = llm_query("Q", "slice")
    assert "[TIME LIMIT REACHED]" in ans
    assert sub.split_calls == []

    # Default (off): an expired deadline does not gate the sub-call — that is the
    # long-standing behavior of the HTTP/vLLM arms and must not change under them.
    sub2 = FakeSplitClient()
    rlm2 = _rlm(sub2)
    rlm2._deadline = time.monotonic() - 1
    llm_query2, _, _ = _llm_query(rlm2)
    assert llm_query2("Q", "slice") == "sub answer"
    assert len(sub2.split_calls) == 1


STUCK_CODE = "```python\nprint(context[:10])\n```"


def test_repetition_breaker_routes_through_chat_split():
    root = FakeClient([STUCK_CODE])
    sub = FakeSplitClient(["the grounded answer"])
    rlm = RLM(root_client=root, sub_client=sub, max_steps=20, exec_timeout=None, run_timeout=None, max_sub_calls=None)

    result = rlm.run("hello world, the capital is Paris", "What is the capital?")
    assert result.end_reason == "repetition_broken"
    assert result.answer == "the grounded answer"
    (call,) = sub.split_calls
    assert "What is the capital?" in call["question"]
    # The compressible side is the material actually seen in the REPL.
    assert "hello worl" in call["context"]
    assert call["system"] == SUB_SYSTEM_PROMPT
    assert result.metrics["sub_split_calls"] == 1


FINISH_CODE = "```python\nFINAL(context[:5])\n```"

# The exact llm_query bullet the dense (HTTP/vLLM) arms run with. Changing it
# re-baselines every arm that reads it, so bump this pin only on purpose.
#
# Generation 2: the cap is rendered from max_subcall_chars instead of the fixed
# "~8000 characters" the text used to claim while enforcement was at 32000 --
# the root was being told to send a quarter of what it was allowed to.
PINNED_DENSE_HELP = """\
    * `llm_query(prompt: str) -> str`: ask a sub-LLM about text. IMPORTANT: the
      sub-LLM CANNOT see `context` or any of your variables — it sees ONLY the
      prompt string you pass. You MUST embed the actual text snippet inside the
      prompt, e.g. llm_query("Answer X based on this text:\\n" + context[i:j])
      (each call is truncated at 32000 characters, so pass a focused
      snippet, not the whole document). Capture the result:
      ans = llm_query(...) then print(ans)."""


def _system_prompt(sub):
    root = FakeClient([FINISH_CODE])
    rlm = RLM(root_client=root, sub_client=sub, max_steps=3, exec_timeout=None, run_timeout=None, max_sub_calls=None)
    result = rlm.run("hello world", "task")
    assert result.finished
    return root.chat_calls[0][0]["content"]


def test_dense_prompt_rendering_is_pinned():
    assert LLM_QUERY_HELP_DENSE.format(max_chars=32000) == PINNED_DENSE_HELP
    prompt = _system_prompt(FakeClient())
    assert PINNED_DENSE_HELP in prompt
    assert EXAMPLE_LLM_QUERY_DENSE in prompt
    assert "context_text" not in prompt


def test_split_prompt_teaches_the_two_arg_form_and_the_actual_cap():
    prompt = _system_prompt(FakeSplitClient())
    assert "llm_query(question: str, context_text: str)" in prompt
    assert "~32000 characters" in prompt  # default max_subcall_chars rendered in
    assert PINNED_DENSE_HELP not in prompt


def test_split_prompt_renders_a_budget_derived_cap():
    """Auto-sizing works by writing the derived size onto max_subcall_chars; the
    prompt has to carry THAT number, or the root keeps asking for slices the
    harness then truncates."""
    root = FakeClient([FINISH_CODE])
    rlm = RLM(
        root_client=root,
        sub_client=FakeSplitClient(),
        max_steps=3,
        exec_timeout=None,
        run_timeout=None,
        max_sub_calls=None,
        max_subcall_chars=108496,
    )
    rlm.run("hello world", "task")
    prompt = root.chat_calls[0][0]["content"]
    assert "~108496 characters" in prompt
    assert "~32000 characters" not in prompt


# --- the slice FLOOR (min_subcall_chars) --------------------------------------
#
# Context for these: across the finished LOFT-128k and LOFT-1m campaigns the root
# sent a mean of 1023 chars per llm_query against a 32000-char cap, and 96% of its
# slices were under 2000 chars. A cap is a ceiling the root may use; the floor is
# what makes the size a property of the harness instead of a property of the
# model's willingness to follow prose.

DOC = "".join(f"paragraph {i} with filler text in it. " for i in range(2000))


def _floored(sub, floor, **kw):
    return _rlm(sub, min_subcall_chars=floor, **kw)


def test_a_keyhole_slice_is_widened_to_the_floor():
    """The dominant shape in the campaigns -- context[max(0, i-500):i+1000] -- is
    exactly what has to stop reaching the sub model at 1500 chars."""
    sub = FakeSplitClient()
    rlm = _floored(sub, 8192)
    llm_query, _, _ = _llm_query(rlm, document=DOC)
    i = DOC.find("paragraph 900")
    llm_query("what?", DOC[max(0, i - 500) : i + 1000])
    assert len(sub.split_calls[0]["context"]) == 8192


def test_a_widened_window_still_contains_what_the_root_asked_about():
    """Widening is only legitimate if it ADDS context around the slice rather than
    relocating it -- otherwise the root's question and the text no longer match."""
    rlm = _floored(FakeSplitClient(), 8192)
    probe = DOC[DOC.find("paragraph 900") :][:40]
    got, expanded = rlm._expand_subcall(probe, DOC)
    assert expanded and probe in got and len(got) == 8192


def test_windows_at_the_document_edges_keep_their_full_width():
    """Symmetric growth clamped at 0 (or at len) yields a SHORT window unless the
    other edge is re-anchored -- and a short window silently misses the floor."""
    rlm = _floored(FakeSplitClient(), 8192)
    for edge in (DOC[:40], DOC[-40:]):
        got, expanded = rlm._expand_subcall(edge, DOC)
        assert expanded and edge in got and len(got) == 8192


def test_a_payload_the_root_assembled_is_left_alone():
    """A string that is not a verbatim slice has no known position in the document,
    so there is no 'around' to widen into. Inventing one would attach unrelated
    text to the root's question."""
    sub = FakeSplitClient()
    rlm = _floored(sub, 8192)
    llm_query, _, _ = _llm_query(rlm, document=DOC)
    llm_query("what?", "text the root built by concatenating fragments")
    assert len(sub.split_calls[0]["context"]) == len("text the root built by concatenating fragments")


def test_a_slice_already_over_the_floor_is_untouched():
    sub = FakeSplitClient()
    rlm = _floored(sub, 8192)
    llm_query, _, _ = _llm_query(rlm, document=DOC)
    llm_query("what?", DOC[:20000])
    assert len(sub.split_calls[0]["context"]) == 20000


def test_the_floor_is_off_by_default_and_changes_no_prompt():
    """Every arm run so far has min_subcall_chars=0; that path must stay identical."""
    sub = FakeSplitClient()
    rlm = _rlm(sub)
    llm_query, _, _ = _llm_query(rlm, document=DOC)
    llm_query("what?", DOC[:50])
    assert len(sub.split_calls[0]["context"]) == 50
    assert rlm._read_instructions(split_capable=True)["example_llm_query"] == EXAMPLE_LLM_QUERY_SPLIT


def test_the_floor_rewrites_help_example_and_strategy_together():
    """All three slots have to agree: the previous split prompt asked for 'FEWER,
    BIGGER slices' one line above an example demonstrating a 700-char keyhole, and
    the example is what the root copied."""
    slots = _floored(FakeSplitClient(), 16384, max_subcall_chars=131072)._read_instructions(split_capable=True)
    assert "at\n      least 16384 characters wide" in slots["llm_query_help"]
    assert "context[max(0, idx-8192):idx+8192]" in slots["example_llm_query"]
    assert "narrow down" not in slots["read_strategy"]
    assert "WHOLE REGION" in slots["read_strategy"]


def test_the_floor_does_not_touch_the_dense_prompt():
    """The large-read text promises a big slice is cheap, which is only true when
    the sub backend compresses it. On a plain chat client that promise is false."""
    slots = _floored(FakeClient(), 16384)._read_instructions(split_capable=False)
    assert slots["read_strategy"] == READ_STRATEGY_DEFAULT
    assert slots["example_llm_query"] == EXAMPLE_LLM_QUERY_DENSE


def test_expansion_is_counted_and_payload_size_is_recorded():
    """runtime.average_sub_payload_chars is the number the chunk-size question turns
    on; sub_call_tokens folds in the question and answer and can only bound it."""
    sub = FakeSplitClient()
    rlm = _floored(sub, 8192)
    llm_query, metrics, _ = _llm_query(rlm, document=DOC)
    llm_query("what?", DOC[:100])
    llm_query("other?", DOC[:20000])
    assert metrics["sub_slices_expanded"] == 1
    assert metrics["sub_payload_chars"] == 8192 + 20000
    assert metrics["sub_payload_chars_max"] == 20000


def test_expansion_is_counted_only_when_the_call_actually_happens():
    """sub_slice_expanded_fraction divides by sub_calls, so counting a widening that
    a cache hit or a budget guard then short-circuits would push it above 1.0."""
    sub = FakeSplitClient()
    rlm = _floored(sub, 8192)
    llm_query, metrics, _ = _llm_query(rlm, document=DOC)
    llm_query("what?", DOC[:100])
    llm_query("what?", DOC[:100])  # same widened window -> served from the cache
    assert metrics["sub_calls"] == 1
    assert metrics["sub_cache_hits"] == 1
    assert metrics["sub_slices_expanded"] == 1


def test_nearby_probes_get_overlapping_windows_each_centred_on_its_own_slice():
    """Windows are centred, so two probes 10 chars apart widen to two DISTINCT
    windows rather than collapsing onto one. Snapping them to a shared grid would
    collapse them, at the cost of letting a hit sit at the very edge of its window
    with its following context cut off -- so centring is the deliberate trade."""
    sub = FakeSplitClient()
    rlm = _floored(sub, 8192)
    llm_query, metrics, _ = _llm_query(rlm, document=DOC)
    i = DOC.find("paragraph 900")
    llm_query("q", DOC[i : i + 50])
    llm_query("q", DOC[i + 10 : i + 60])
    assert len(sub.split_calls) == 2
    assert metrics["sub_cache_hits"] == 0
    first, second = (c["context"] for c in sub.split_calls)
    assert first != second
    assert DOC[i : i + 50] in first and DOC[i + 10 : i + 60] in second
    # Centred, so each probe sits near the middle rather than at an edge.
    assert 0.4 < first.find(DOC[i : i + 50]) / len(first) < 0.6
