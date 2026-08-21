# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-process kvpress sub-client used by the RLM's arm 4.

Runs on CPU with the 0-size unit-test llama so the duck-type surface, the stats
accounting, and the press wiring are exercised without a GPU. The full-size
behavior (KVzip on 32k-token slices) is covered by the smoke steps in
evaluation/rlm/loft128k/README.md, not here.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kvpress")

from evaluation.rlm.kvpress_backend import KVPressSubClient, build_press  # noqa: E402

TINY_MODEL = "MaxJeblick/llama2-0b-unit-test"


def test_build_press_maps_names_and_propagates_the_ratio():
    from kvpress import KVzipPress, SnapKVPress

    assert build_press("no_press", 0.5) is None
    press = build_press("kvzip", 0.75)
    assert isinstance(press, KVzipPress)
    assert press.compression_ratio == 0.75
    assert isinstance(build_press("snapkv", 0.25), SnapKVPress)
    with pytest.raises(ValueError, match="unknown press"):
        build_press("definitely_not_a_press", 0.5)


@pytest.fixture(scope="module")
def client():
    return KVPressSubClient(
        model=TINY_MODEL,
        press_name="snapkv",
        compression_ratio=0.25,
        device="cpu",
        max_new_tokens=8,
        press_min_tokens=8,
    )


def test_duck_type_surface(client):
    assert client.model == TINY_MODEL
    client.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}  # assignable no-op
    assert client.usage.total_tokens == 0
    assert client.usage.calls == 0


def test_chat_split_generates_and_accounts(client):
    # Long enough to clear both press_min_tokens and SnapKV's 64-token window.
    context = "The quick brown fox jumps over the lazy dog. " * 12
    answer = client.chat_split(question="What jumps?", context=context, system="Answer briefly.")

    assert isinstance(answer, str)
    assert client.usage.calls == 1
    assert client.usage.prompt_tokens > 0
    assert client.usage.completion_tokens > 0

    stats = client.pop_example_stats()
    assert stats["calls"] == 1
    assert stats["split_calls"] == 1
    assert stats["pressed_calls"] == 1
    assert stats["average_context_tokens"] > 0
    # Ratio path: retained = (1 - ratio) x context, reported by the pipeline.
    assert stats["average_retained_context_tokens"] < stats["average_context_tokens"]
    assert stats["average_compression_ratio"] == pytest.approx(0.25)
    # The buffer is per-example: a second pop must be empty.
    assert client.pop_example_stats() == {}


def test_one_arg_chat_below_press_min_tokens_stays_dense(client):
    answer = client.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    assert isinstance(answer, str)
    stats = client.pop_example_stats()
    assert stats["calls"] == 1
    assert stats["split_calls"] == 0
    assert stats["pressed_calls"] == 0
    assert stats["average_compression_ratio"] == 0.0


def test_one_arg_chat_above_press_min_tokens_is_compressed(client):
    big = "word " * 200  # ~200 tokens > press_min_tokens=8
    client.chat([{"role": "user", "content": big}])
    stats = client.pop_example_stats()
    assert stats["pressed_calls"] == 1


def test_short_split_context_skips_the_press(client):
    # Below press_min_tokens even on the split path: some presses (SnapKV's
    # 64-token window) crash outright on tiny contexts, so the floor applies to
    # both call forms and the skip is visible in the stats.
    client.chat_split(question="Q", context="tiny")
    stats = client.pop_example_stats()
    assert stats["split_calls"] == 1
    assert stats["pressed_calls"] == 0


def test_kvzip_smoke_on_a_tiny_context():
    kv = KVPressSubClient(
        model=TINY_MODEL,
        press_name="kvzip",
        compression_ratio=0.5,
        device="cpu",
        max_new_tokens=4,
        press_min_tokens=8,
    )
    answer = kv.chat_split(question="What is here?", context="Some short context here. " * 30)
    assert isinstance(answer, str)
    assert kv.pop_example_stats()["pressed_calls"] == 1
