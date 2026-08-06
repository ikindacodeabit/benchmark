# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the properties that make an RLM run comparable to a KVPress run.

Run from the repository root (the module uses absolute `evaluation.*` imports):
    python -m pytest evaluation/rlm/test_run_benchmark.py
"""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.rlm.rlm import vanilla_answer
from evaluation.rlm.run_benchmark import load_done, write_run_artifacts


def _checkpoint(records: list[dict]) -> tuple[Path, Path]:
    run_dir = Path(tempfile.mkdtemp())
    path = run_dir / "checkpoint.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path, run_dir


def _answered(example_id: str, answer: str, pred: str, peak: int = 4096) -> dict:
    return dict(
        id=example_id,
        task="32k",
        answer=answer,
        pred=pred,
        correct=answer == pred,
        tokens=10,
        latency_s=1.0,
        finished=True,
        metrics={"peak_context_tokens": peak},
    )


class ErroredRecordTest(unittest.TestCase):
    """An API failure is a missing measurement, not a wrong answer.

    KVPress runs locally and never makes network calls, so it cannot take this
    hit. Counting a NIM outage as a zero would show up as the RLM arm being less
    accurate, which is the one thing this comparison must not get wrong.
    """

    records = [
        _answered("a", "alpha", "alpha"),
        _answered("b", "beta", "beta"),
        _answered("c", "gamma", "wrong"),
        dict(
            id="d",
            task="32k",
            answer="delta",
            pred=None,
            correct=False,
            tokens=0,
            latency_s=0.5,
            finished=False,
            error="APIError: 503 Service Unavailable",
        ),
    ]

    def test_error_is_excluded_from_the_score_but_still_reported(self):
        path, run_dir = _checkpoint(self.records)
        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})

        # 2 of the 3 ANSWERED examples are right. Scoring the error as "" would
        # give 50.0 and understate the arm by 16.7 points.
        self.assertAlmostEqual(metrics["32k"]["string_match"], 200 / 3, places=1)
        self.assertEqual(metrics["32k"]["num_samples"], 3)
        self.assertEqual(metrics["runtime"]["scored"], 3)
        self.assertEqual(metrics["runtime"]["errors"], 1)
        self.assertEqual(metrics["runtime"]["examples"], 4)

    def test_errored_example_is_retried_on_resume(self):
        path, _ = _checkpoint(self.records)
        self.assertEqual(load_done(path), {"a", "b", "c"})

    def test_peak_context_tokens_is_reported_for_the_cost_axis(self):
        path, run_dir = _checkpoint([_answered("a", "x", "x", peak=4000), _answered("b", "y", "y", peak=4200)])
        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})
        self.assertAlmostEqual(metrics["runtime"]["average_peak_context_tokens"], 4100.0)


class _ContextOverflow(Exception):
    status_code = 400

    def __str__(self):
        return "This model's maximum context length is 40960 tokens"


class _BadRequest(Exception):
    status_code = 400

    def __str__(self):
        return "invalid model id"


class _FakeClient:
    """Rejects prompts over `limit` characters, as vLLM rejects them over tokens."""

    def __init__(self, limit: int, error=_ContextOverflow):
        self.limit = limit
        self.error = error
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if len(messages[0]["content"]) > self.limit:
            raise self.error()
        return "the answer"


class VanillaTruncationTest(unittest.TestCase):
    """The vanilla arm's ceiling has to be visible, and self-inflicted.

    A character limit is not a token limit, so on densely-tokenising subsets the
    server used to reject every request; those cells then scored 0.0 from a
    harness error rather than from the model.
    """

    context = "x" * 200_000

    def test_shrinks_and_retries_until_the_server_accepts_the_prompt(self):
        client = _FakeClient(limit=50_000)
        stats: dict = {}
        answer = vanilla_answer(client, self.context, "find it", char_limit=400_000, stats=stats)

        self.assertEqual(answer, "the answer")
        self.assertGreater(client.calls, 1)
        self.assertTrue(stats["truncated"])
        self.assertLessEqual(stats["context_chars_used"], 50_000)

    def test_records_truncation_even_when_the_first_call_succeeds(self):
        client = _FakeClient(limit=10**9)
        stats: dict = {}
        vanilla_answer(client, self.context, "find it", char_limit=100_000, stats=stats)

        self.assertEqual(client.calls, 1)
        self.assertEqual(stats["context_chars"], 200_000)
        self.assertEqual(stats["context_chars_used"], 100_000)
        self.assertTrue(stats["truncated"])

    def test_a_non_length_400_is_not_swallowed_by_the_retry_loop(self):
        client = _FakeClient(limit=0, error=_BadRequest)
        with self.assertRaises(_BadRequest):
            vanilla_answer(client, self.context, "find it", stats={})
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
