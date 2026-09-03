# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the properties that make an RLM run comparable to a KVPress run.

Run from the repository root (the module uses absolute `evaluation.*` imports):
    python -m pytest tests/evaluation/test_rlm_run_benchmark.py
"""

import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from evaluation.rlm import run_benchmark as rb
from evaluation.rlm.rlm import Scratchpad, vanilla_answer
from evaluation.rlm.run_benchmark import (
    build_run_dir_components,
    load_done,
    resolve_compression_target,
    resume_conflicts,
    write_run_artifacts,
)


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
    hit. Counting a server outage as a zero would show up as the RLM arm being less
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

    def test_a_retried_error_does_not_count_twice(self):
        """The failed record stays in the append-only checkpoint when the example
        is retried, so without de-duplication the run reports an error it already
        recovered from -- and `errors` is the column that vets the score."""
        retried = self.records + [_answered("d", "delta", "delta")]
        path, run_dir = _checkpoint(retried)
        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})

        self.assertEqual(metrics["runtime"]["errors"], 0)
        self.assertEqual(metrics["runtime"]["examples"], 4)
        self.assertEqual(metrics["runtime"]["scored"], 4)
        predictions = (run_dir / "predictions.csv").read_text().splitlines()
        self.assertEqual(sum(line.startswith("d,") for line in predictions), 1)

    def test_abstentions_are_counted_apart_from_failures(self):
        records = [_answered("a", "x", "x"), _answered("b", "y", "y")]
        records[1].update(pred=None, end_reason="abstained", correct=False)
        records[0]["end_reason"] = "final_called"
        path, run_dir = _checkpoint(records)
        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})
        self.assertEqual(metrics["runtime"]["abstained"], 1)


class ResumeGuardTest(unittest.TestCase):
    """The run-dir name cannot carry every result-affecting knob, so resuming is
    checked against the config the first run wrote. Two experiments must never
    merge into one checkpoint.jsonl."""

    @staticmethod
    def _run_dir(**config) -> Path:
        run_dir = Path(tempfile.mkdtemp())
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
        return run_dir

    def test_a_changed_limit_is_a_conflict(self):
        run_dir = self._run_dir(limit=110, max_steps=50)
        conflicts = resume_conflicts(run_dir, {"limit": 10, "max_steps": 50})
        self.assertEqual(len(conflicts), 1)
        self.assertIn("limit", conflicts[0])

    def test_matching_configuration_resumes_cleanly(self):
        run_dir = self._run_dir(limit=110, max_steps=50, sub_model="a/b")
        self.assertEqual(resume_conflicts(run_dir, {"limit": 110, "max_steps": 50, "sub_model": "a/b"}), [])

    def test_fixed_grid_axes_are_resume_critical(self):
        run_dir = self._run_dir(fixed_chunk=True, compression_factor=8)
        conflicts = resume_conflicts(run_dir, {"fixed_chunk": True, "compression_factor": 16})
        self.assertEqual(len(conflicts), 1)
        self.assertIn("compression_factor", conflicts[0])

    def test_the_sample_is_resume_critical(self):
        """--sample-fraction reaches neither the run-dir name nor the slug, so the
        config guard is the only thing standing between two different samples and
        one shared checkpoint.jsonl."""
        run_dir = self._run_dir(limit=20, sample_fraction=0.5, sample_seed=42)
        self.assertEqual(resume_conflicts(run_dir, {"limit": 20, "sample_fraction": 0.5, "sample_seed": 42}), [])
        for changed in ({"sample_fraction": 1.0}, {"sample_seed": 7}):
            conflicts = resume_conflicts(run_dir, {"limit": 20, "sample_fraction": 0.5, "sample_seed": 42, **changed})
            self.assertEqual(len(conflicts), 1, changed)
            self.assertIn(next(iter(changed)), conflicts[0])

    def test_runs_predating_the_sample_flags_still_resume(self):
        """Every config.yaml already on disk lacks these keys; the guard reports
        what the previous run actually recorded, so they must not become conflicts."""
        run_dir = self._run_dir(limit=20, max_steps=50)
        self.assertEqual(
            resume_conflicts(run_dir, {"limit": 20, "max_steps": 50, "sample_fraction": None, "sample_seed": None}), []
        )

    def test_a_missing_config_does_not_block_a_resume(self):
        """Checkpoints written before config.yaml was saved up front are still
        resumable -- the guard reports what it can prove, not what it assumes."""
        self.assertEqual(resume_conflicts(Path(tempfile.mkdtemp()), {"limit": 10}), [])

    def test_knobs_outside_the_declared_set_are_not_conflicts(self):
        run_dir = self._run_dir(limit=110, base_url="http://a")
        self.assertEqual(resume_conflicts(run_dir, {"limit": 110, "base_url": "http://b"}), [])

    def test_the_guard_reads_the_previous_run_not_this_one(self):
        """config.yaml is written up front so an interrupted run is still checkable
        -- which means the write must come AFTER the checks read it. Writing it any
        earlier makes every guard compare this run against itself and always pass."""
        source = inspect.getsource(rb.main)
        write = source.index('(run_dir / "config.yaml").write_text')
        self.assertLess(source.index("conflicts = resume_conflicts("), write)
        self.assertLess(source.index("prior = _prior_resolved_chars("), write)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        dataset="loft",
        legacy_dataset=None,
        data_dir="nq_128k",
        root_model="Qwen/Qwen3-4B-Instruct-2507",
        split="all",
        max_context_tokens=None,
        sub_backend="http",
        press="kvzip",
        memory_budget=1.0,
        memory_budget_unit="GB",
        max_subcall_chars=32000,
        subcall_sizing_mode="fixed",
        target_compression_ratio=None,
        compression_factor=None,
        fixed_chunk=False,
        min_subcall_chars=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class RunDirComponentsTest(unittest.TestCase):
    """Anything that changes results must land in the directory name, or two
    configurations share a checkpoint.jsonl and silently merge."""

    def test_http_backend_keeps_the_legacy_layout(self):
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

    def test_auto_sizing_directory_ignores_the_resolved_size(self):
        """main() overwrites max_subcall_chars with the resolved int BEFORE the
        run dir is named. If the name carried it, two auto runs resolving to
        different sizes (GPU free memory changed) would land in different
        directories and silently fork instead of hitting the resume guard."""
        auto = dict(sub_backend="kvzip", subcall_sizing_mode="auto", target_compression_ratio=0.9)
        first = build_run_dir_components(_args(**auto, max_subcall_chars=101000), "rlm", None)
        second = build_run_dir_components(_args(**auto, max_subcall_chars=99000), "rlm", None)
        self.assertEqual(first, second)

    def test_fixed_factor_grid_has_human_readable_axes(self):
        args = _args(
            dataset="longbench128k",
            data_dir="narrativeqa",
            sub_backend="kvzip",
            press="kvzip",
            memory_budget=8192,
            memory_budget_unit="tokens",
            subcall_sizing_mode="auto",
            target_compression_ratio=0.9375,
            compression_factor=16,
            fixed_chunk=True,
            max_subcall_chars=769047,
            min_subcall_chars=572785,
        )
        self.assertEqual(
            build_run_dir_components(args, "rlm", Scratchpad()),
            [
                "longbench128k",
                "narrativeqa",
                "Qwen_Qwen3-4B-Instruct-2507",
                "rlm",
                "scratchpad",
                "kvzip-kvzip8192tokens",
                "autosubx16",
                "fixed",
            ],
        )

    def test_a_split_filter_gets_its_own_directory(self):
        """A dev-split smoke run must not resume into the real run's checkpoint --
        but `all` adds nothing, so existing directories keep their names."""
        self.assertEqual(
            build_run_dir_components(_args(), "rlm", None)[:4],
            ["loft", "nq_128k", "Qwen_Qwen3-4B-Instruct-2507", "rlm"],
        )
        self.assertIn("split-dev", build_run_dir_components(_args(split="dev"), "rlm", None))
        self.assertNotIn("split-all", build_run_dir_components(_args(), "rlm", None))

    def test_vanilla_mode_ignores_rlm_only_knobs(self):
        args = _args(sub_backend="kvzip", max_subcall_chars=131072, max_context_tokens=4096)
        self.assertEqual(
            build_run_dir_components(args, "vanilla", Scratchpad()),
            ["loft", "nq_128k", "Qwen_Qwen3-4B-Instruct-2507", "vanilla"],
        )


class FixedChunkFlagTest(unittest.TestCase):
    def test_factor_is_converted_to_the_existing_ratio_axis(self):
        args = _args(
            sub_backend="kvzip",
            max_subcall_chars="auto",
            compression_factor=8,
            fixed_chunk=True,
        )
        self.assertEqual(resolve_compression_target(args), 0.875)

    def test_fixed_chunk_requires_auto_kvzip_and_owns_its_floor(self):
        with self.assertRaisesRegex(ValueError, "requires --sub-backend kvzip"):
            resolve_compression_target(_args(fixed_chunk=True, compression_factor=2))
        with self.assertRaisesRegex(ValueError, "derives its own floor"):
            resolve_compression_target(
                _args(
                    sub_backend="kvzip",
                    max_subcall_chars="auto",
                    compression_factor=2,
                    fixed_chunk=True,
                    min_subcall_chars=100,
                )
            )

    def test_factor_and_ratio_are_mutually_exclusive_even_for_programmatic_callers(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            resolve_compression_target(
                _args(
                    sub_backend="kvzip",
                    max_subcall_chars="auto",
                    compression_factor=2,
                    target_compression_ratio=0.5,
                )
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
                    "context_tokens_on_target_calls": 4,
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
                    "context_tokens_on_target_calls": 1,
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
        self.assertAlmostEqual(runtime["realized_compression_factor"], 2.0)
        self.assertAlmostEqual(runtime["sub_context_tokens_on_target_fraction"], 5 / 6)

    def test_runs_without_sub_kv_report_nothing(self):
        path, run_dir = _checkpoint([_answered("a", "x", "x")])
        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})
        self.assertNotIn("average_sub_retained_context_tokens", metrics["runtime"])

    def test_fixed_read_health_metrics_are_aggregated(self):
        records = [_answered("a", "x", "x"), _answered("b", "x", "x")]
        records[0]["metrics"].update(document_coverage_fraction=0.75, sub_slice_unlocatable_calls=2)
        records[1]["metrics"].update(document_coverage_fraction=0.25, sub_slice_unlocatable_calls=1)
        path, run_dir = _checkpoint(records)

        metrics = write_run_artifacts(path, run_dir, "synthetic_kv_32k", {"backend": "rlm"})

        self.assertEqual(metrics["runtime"]["sub_slice_unlocatable_calls"], 3)
        self.assertEqual(metrics["runtime"]["document_coverage_fraction"], 0.5)


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
