# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the properties that make an RLM run comparable to a KVPress run.

Run from the repository root (the module uses absolute `evaluation.*` imports):
    python -m pytest evaluation/rlm/test_run_benchmark.py
"""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.rlm.rlm import Scratchpad, vanilla_answer
from evaluation.rlm.run_benchmark import build_run_dir_components, load_done, write_run_artifacts


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


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        dataset="loft",
        legacy_dataset=None,
        data_dir="nq_128k",
        root_model="Qwen/Qwen3-4B-Instruct-2507",
        max_context_tokens=None,
        sub_backend="nim",
        press="kvzip",
        memory_budget=1.0,
        memory_budget_unit="GB",
        max_subcall_chars=32000,
        subcall_sizing_mode="fixed",
        target_compression_ratio=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class RunDirComponentsTest(unittest.TestCase):
    """Anything that changes results must land in the directory name, or two
    configurations share a checkpoint.jsonl and silently merge."""

    def test_nim_backend_keeps_the_legacy_layout(self):
        self.assertEqual(
            build_run_dir_components(_args(), "rlm", Scratchpad()),
            ["loft", "nq_128k", "Qwen_Qwen3-4B-Instruct-2507", "rlm", "scratchpad"],
        )

    def test_kvzip_backend_stamps_press_budget_and_nondefault_chunk(self):
        args = _args(sub_backend="kvzip", memory_budget=2.0, max_subcall_chars=131072)
        self.assertEqual(
            build_run_dir_components(args, "rlm", Scratchpad()),
            ["loft", "nq_128k", "Qwen_Qwen3-4B-Instruct-2507", "rlm", "scratchpad", "kvzip-kvzip2GB", "sub131072"],
        )

    def test_default_chunk_size_adds_no_suffix(self):
        args = _args(sub_backend="kvzip")
        self.assertEqual(
            build_run_dir_components(args, "rlm", None)[-1],
            "kvzip-kvzip1GB",
        )

    def test_auto_sizing_gets_its_own_directory(self):
        """An auto run that happens to resolve to exactly the default size must NOT
        resume into the hand-sized checkpoint: the marker is the only thing
        distinguishing them, since neither carries a `sub<N>` component."""
        fixed = build_run_dir_components(_args(sub_backend="kvzip"), "rlm", None)
        auto = build_run_dir_components(
            _args(sub_backend="kvzip", subcall_sizing_mode="auto", target_compression_ratio=0.9),
            "rlm",
            None,
        )
        self.assertEqual(auto[-1], "autosub0.9")
        self.assertNotIn("sub32000", auto)
        self.assertNotEqual(fixed, auto)

    def test_vanilla_mode_ignores_rlm_only_knobs(self):
        args = _args(sub_backend="kvzip", max_subcall_chars=131072, max_context_tokens=4096)
        self.assertEqual(
            build_run_dir_components(args, "vanilla", Scratchpad()),
            ["loft", "nq_128k", "Qwen_Qwen3-4B-Instruct-2507", "vanilla"],
        )


class SubKvAggregationTest(unittest.TestCase):
    """The run-level sub-side retention stats are the RLM analogue of KVPress's
    average_retained_context_tokens; near-zero split fraction is the tell that
    the arm degenerated to dense one-arg calls."""

    @staticmethod
    def _with_sub_kv(example_id: str, sub_kv: dict | None) -> dict:
        record = _answered(example_id, "x", "x")
        record["metrics"] = {"peak_context_tokens": 4096, "sub_kv": sub_kv}
        return record

    def test_sub_kv_is_averaged_into_runtime(self):
        records = [
            self._with_sub_kv(
                "a",
                {
                    "calls": 4,
                    "split_calls": 3,
                    "average_context_tokens": 8000.0,
                    "average_retained_context_tokens": 4000.0,
                    "average_compression_ratio": 0.5,
                },
            ),
            self._with_sub_kv(
                "b",
                {
                    "calls": 2,
                    "split_calls": 0,
                    "average_context_tokens": 6000.0,
                    "average_retained_context_tokens": 3000.0,
                    "average_compression_ratio": 0.5,
                },
            ),
            self._with_sub_kv("c", None),  # an example whose run made no sub-calls
        ]
        path, run_dir = _checkpoint(records)
        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})

        runtime = metrics["runtime"]
        self.assertAlmostEqual(runtime["average_sub_context_tokens"], 7000.0)
        self.assertAlmostEqual(runtime["average_sub_retained_context_tokens"], 3500.0)
        self.assertAlmostEqual(runtime["average_sub_compression_ratio"], 0.5)
        self.assertAlmostEqual(runtime["sub_split_call_fraction"], 0.5)

    def test_runs_without_sub_kv_report_nothing(self):
        path, run_dir = _checkpoint([_answered("a", "x", "x")])
        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})
        self.assertNotIn("average_sub_retained_context_tokens", metrics["runtime"])


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
