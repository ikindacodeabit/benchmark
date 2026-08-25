# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the split (question, context) llm_query path used by the kvpress
sub-backend.

A split-capable sub client receives the context slice as a separate argument so
the press can compress it apart from the question. These tests pin three things:
the dispatch contract, the fallback behavior on plain clients, and — most
importantly — that the DENSE prompt rendering used by the in-flight NIM/vLLM
arms did not change (a wording change mid-campaign would confound them).
"""

import time

from evaluation.rlm.rlm import EXAMPLE_LLM_QUERY_DENSE, LLM_QUERY_HELP_DENSE, RLM, SUB_SYSTEM_PROMPT


class FakeClient:
    """Plain NIMClient stand-in: chat only, records calls."""

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


def _llm_query(rlm, metrics=None, cache=None):
    metrics = _metrics() if metrics is None else metrics
    cache = {} if cache is None else cache
    env = rlm._make_env("the full document", metrics, cache, [])
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
    assert len(call["context"]) == 100
    assert "[NOTE: your prompt was truncated at 100 chars" in call["question"]
    assert "[NOTE:" not in call["context"]


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
    # long-standing behavior of the NIM/vLLM arms and must not change under them.
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

# The exact llm_query bullet the NIM/vLLM arms have been running with. If this
# assertion fails, the change would put arms 2/3 mid-campaign on a different
# prompt than their existing checkpoints — bump this pin only on purpose.
PINNED_DENSE_HELP = """\
    * `llm_query(prompt: str) -> str`: ask a sub-LLM about text. IMPORTANT: the
      sub-LLM CANNOT see `context` or any of your variables — it sees ONLY the
      prompt string you pass. You MUST embed the actual text snippet inside the
      prompt, e.g. llm_query("Answer X based on this text:\\n" + context[i:j])
      (keep each call under ~8000 characters). Capture the result:
      ans = llm_query(...) then print(ans)."""


def _system_prompt(sub):
    root = FakeClient([FINISH_CODE])
    rlm = RLM(root_client=root, sub_client=sub, max_steps=3, exec_timeout=None, run_timeout=None, max_sub_calls=None)
    result = rlm.run("hello world", "task")
    assert result.finished
    return root.chat_calls[0][0]["content"]


def test_dense_prompt_rendering_is_pinned():
    assert LLM_QUERY_HELP_DENSE == PINNED_DENSE_HELP
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
