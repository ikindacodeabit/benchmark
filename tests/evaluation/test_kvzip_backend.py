# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the GPU/dependency preflight checks in kvzip_backend.

Everything here runs without a GPU or torch: query_gpus is monkeypatched, so
the tests exercise the selection/error logic itself — the part that must fail
FAST and CLEARLY instead of OOMing twenty minutes into a run.
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from evaluation.rlm import kvzip_backend


def _gpus(*free_gib):
    return [{"index": i, "free_gib": f, "total_gib": 48.0} for i, f in enumerate(free_gib)]


class PreflightSelectGpuTest(unittest.TestCase):
    def setUp(self):
        # Each test starts unpinned; restore whatever the shell had afterwards.
        self._saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._saved

    def test_picks_the_freest_gpu_and_pins_it(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(5.0, 30.0, 20.0)):
            chosen = kvzip_backend.preflight_select_gpu(min_free_gib=14.0)
        self.assertEqual(chosen, 1)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_honors_an_explicit_device_when_it_has_room(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(20.0, 30.0)):
            chosen = kvzip_backend.preflight_select_gpu(min_free_gib=14.0, device="cuda:0")
        self.assertEqual(chosen, 0)

    def test_explicit_device_without_room_fails_with_the_per_gpu_picture(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(5.0, 30.0)):
            with self.assertRaisesRegex(RuntimeError, r"GPU 0 has 5\.0 GiB free.*GPU1: 30\.0"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0, device="cuda:0")

    def test_pre_set_cuda_visible_devices_is_respected_not_overridden(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(40.0, 40.0, 20.0, 5.0)):
            chosen = kvzip_backend.preflight_select_gpu(min_free_gib=14.0)
        self.assertEqual(chosen, 2)

    def test_no_gpu_has_room(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(5.0, 8.0)):
            with self.assertRaisesRegex(RuntimeError, "No GPU has"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0)

    def test_no_nvidia_gpu_at_all(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "No NVIDIA GPU visible"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0)

    def test_nonexistent_pinned_gpu(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(30.0)):
            with self.assertRaisesRegex(RuntimeError, "GPU 7 does not exist"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0, device="cuda:7")


def _fake_client(free_bytes=40 * 1024**3, max_position_embeddings=262144, max_context_tokens=34000):
    """A KVzipSubClient with the model/GPU dependencies stubbed out.

    Built via __new__ so no weights load and no CUDA context is created — the
    same trick tests/test_pipeline.py uses. Config numbers are
    Qwen3-4B-Instruct-2507's, so kv_bytes_per_token comes out at the real 147,456.
    """
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
            max_position_embeddings=max_position_embeddings,
        ),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=object()) for _ in range(36)]),
        dtype=torch.bfloat16,
    )
    # 1 char == 1 token, so the calibrated ratio is a clean 1.0.
    tokenizer = SimpleNamespace(encode=lambda s, add_special_tokens=False: list(range(len(s))))

    client = kvzip_backend.KVzipSubClient.__new__(kvzip_backend.KVzipSubClient)
    client.pipeline = SimpleNamespace(model=model, tokenizer=tokenizer)  # type: ignore[assignment]
    client.memory_budget = 1.0
    client.memory_budget_unit = "GB"
    client.max_context_tokens = max_context_tokens
    client._kv_bytes_per_token = 36 * 2 * 8 * 128 * 2
    client._torch = SimpleNamespace(  # type: ignore[assignment]
        cuda=SimpleNamespace(mem_get_info=lambda: (free_bytes, free_bytes))
    )
    client.subcall_sizing = None
    return client


class PlanSubcallChunkTest(unittest.TestCase):
    """The planner's plumbing: budget in, size out, every cap considered."""

    def test_it_derives_the_size_from_the_budget_and_records_the_reasoning(self):
        client = _fake_client(max_context_tokens=131072)
        sizing = client.plan_subcall_chunk(document="word " * 20000, target_compression_ratio=0.75)

        # 1 GB / 147,456 B per token = 6,781 retained; /(1-0.75) = 27,124 admitted.
        self.assertEqual(sizing.token_budget, 6781)
        self.assertEqual(sizing.tokens, 27124)
        self.assertEqual(sizing.binding, "budget")
        self.assertAlmostEqual(sizing.realized_ratio_if_filled, 0.75, places=3)
        self.assertEqual(sizing.kv_bytes_per_token, 147456)
        self.assertIs(client.subcall_sizing, sizing)

    def test_the_cli_context_cap_is_applied_when_none_is_passed(self):
        """--sub-max-context-tokens must bind even though the caller didn't repeat it."""
        client = _fake_client(max_context_tokens=34000)
        sizing = client.plan_subcall_chunk(document="word " * 20000, target_compression_ratio=0.9)
        self.assertEqual(sizing.binding, "cli_cap")
        self.assertLess(sizing.realized_ratio_if_filled, 0.9)

    def test_a_full_gpu_binds_the_size(self):
        client = _fake_client(free_bytes=3 * 1024**3, max_context_tokens=131072)
        sizing = client.plan_subcall_chunk(document="word " * 20000, target_compression_ratio=0.9)
        self.assertEqual(sizing.binding, "gpu_fit")

    def test_the_model_window_is_read_from_the_config(self):
        client = _fake_client(max_position_embeddings=8192, max_context_tokens=131072)
        sizing = client.plan_subcall_chunk(document="word " * 20000, target_compression_ratio=0.9)
        self.assertEqual(sizing.binding, "sub_window")
        self.assertEqual(sizing.caps["sub_window"], 8192 - 1024)

    def test_the_reserve_reaches_the_planner(self):
        """--subcall-reserve-tokens has to actually move the caps, or the question
        and the decoded answer are unbudgeted."""
        client = _fake_client(max_position_embeddings=8192, max_context_tokens=131072)
        sizing = client.plan_subcall_chunk(document="word " * 20000, target_compression_ratio=0.9, reserve_tokens=4096)
        self.assertEqual(sizing.caps["sub_window"], 8192 - 4096)
        self.assertEqual(sizing.tokens, 8192 - 4096)

    def test_the_min_token_floor_reaches_the_planner(self):
        """The small-budget grid rows exist only if --subcall-min-tokens is honoured.

        A B=128 x F=4 cell resolves to N=512, which the 1024 default refuses. That
        default dates from a fixed 1024-token press floor; press_min_tokens is now
        derived from the budget, so the cell is legitimate and must be runnable.
        """
        client = _fake_client(max_context_tokens=131072)
        client.memory_budget = 128
        client.memory_budget_unit = "tokens"

        with self.assertRaises(RuntimeError):
            client.plan_subcall_chunk(document="word " * 20000, target_compression_ratio=0.75)

        sizing = client.plan_subcall_chunk(
            document="word " * 20000,
            target_compression_ratio=0.75,
            min_tokens=128,
        )
        self.assertEqual(sizing.tokens, 512)
        self.assertEqual(sizing.binding, "budget")

    def test_fixed_grid_ignores_legacy_cli_cap_but_keeps_hard_caps_strict(self):
        client = _fake_client(max_context_tokens=34_000)
        client.memory_budget = 8192
        client.memory_budget_unit = "tokens"
        sizing = client.plan_subcall_chunk(
            document="word " * 20000,
            target_compression_ratio=0.9375,
            char_overshoot=1.15,
            require_budget_binding=True,
            apply_cli_context_cap=False,
        )
        self.assertEqual(sizing.tokens, 131_072)
        self.assertEqual(sizing.binding, "budget")
        self.assertIsNone(sizing.caps["cli_cap"])


class TruncationAccountingTest(unittest.TestCase):
    def test_decode_reencode_drift_is_measured_instead_of_recording_the_cap(self):
        class DriftTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text)))

            def decode(self, ids):
                return "x" * (len(ids) - 1)

        client = kvzip_backend.KVzipSubClient.__new__(kvzip_backend.KVzipSubClient)
        client.pipeline = SimpleNamespace(tokenizer=DriftTokenizer())  # type: ignore[assignment]
        client.max_context_tokens = 5

        text, measured, dropped = client._truncate_context_to_token_cap("abcdefghij")

        self.assertEqual(text, "xxxx")
        self.assertEqual(measured, 4)
        self.assertEqual(dropped, 5)


if __name__ == "__main__":
    unittest.main()
