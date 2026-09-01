# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deriving the RLM sub-call chunk size from a KV memory budget.

Lives under `tests/` rather than beside `evaluation/rlm/test_*.py` on purpose:
the Makefile's `test` target only collects `tests/`, so anything next to the
other RLM tests never runs in CI.

Pure -- no torch, no GPU, no model. The reference numbers throughout are
Qwen3-4B-Instruct-2507: 36 layers x 2 (K+V) x 8 KV heads x 128 head_dim x 2
bytes = 147,456 bytes/token, which is the "144 KiB per token" quoted in
evaluation/rlm/loft128k/README.md.
"""

import unittest

from evaluation.rlm.sizing import (
    CHARS_PER_TOKEN_BOUNDS,
    DEFAULT_CHARS_PER_TOKEN,
    FIXED_CHUNK_CHAR_OVERSHOOT,
    KV_FIT_HEADROOM_BYTES,
    KV_FIT_SAFETY_FACTOR,
    calibrate_chars_per_token,
    compression_ratio_from_factor,
    fixed_chunk_char_cap,
    gpu_fit_token_cap,
    size_subcall_chunk,
)

QWEN3_4B_KV_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2  # 147,456
ONE_GB_TOKEN_BUDGET = 1_000_000_000 // QWEN3_4B_KV_BYTES_PER_TOKEN  # 6,781


def _size(**overrides):
    base = dict(
        token_budget=ONE_GB_TOKEN_BUDGET,
        target_compression_ratio=0.9,
        chars_per_token=4.0,
        kv_bytes_per_token=QWEN3_4B_KV_BYTES_PER_TOKEN,
        memory_budget=1.0,
        memory_budget_unit="GB",
    )
    base.update(overrides)
    return size_subcall_chunk(**base)


class BudgetInversionTest(unittest.TestCase):
    def test_factor_maps_to_the_grid_ratio(self):
        for factor in (1, 2, 4, 8, 16):
            self.assertEqual(compression_ratio_from_factor(factor), 1 - 1 / factor)
        with self.assertRaises(ValueError):
            compression_ratio_from_factor(0.5)

    def test_budget_binds_when_no_other_cap_is_supplied(self):
        sizing = _size()
        self.assertEqual(sizing.tokens, ONE_GB_TOKEN_BUDGET // 1 * 10)  # 6781 / 0.1
        self.assertEqual(sizing.binding, "budget")
        self.assertEqual(sizing.chars, sizing.tokens * 4)

    def test_a_zero_target_asks_for_exactly_the_budget(self):
        """target 0 means 'a chunk that fits uncompressed' -- the press does nothing."""
        sizing = _size(target_compression_ratio=0.0)
        self.assertEqual(sizing.tokens, ONE_GB_TOKEN_BUDGET)
        self.assertEqual(sizing.realized_ratio_if_filled, 0.0)

    def test_realized_ratio_lands_on_the_target_when_the_budget_binds(self):
        sizing = _size(target_compression_ratio=0.75)
        self.assertAlmostEqual(sizing.realized_ratio_if_filled, 0.75, places=3)

    def test_clamping_pushes_the_realized_ratio_below_the_target(self):
        """The number that matters: a clamped chunk cannot reach the asked-for ratio."""
        sizing = _size(target_compression_ratio=0.9, cli_max_context_tokens=34_000)
        self.assertEqual(sizing.binding, "cli_cap")
        self.assertLess(sizing.realized_ratio_if_filled, 0.9)
        self.assertAlmostEqual(sizing.realized_ratio_if_filled, 1 - ONE_GB_TOKEN_BUDGET / sizing.tokens)

    def test_target_ratio_out_of_range_is_rejected(self):
        for bad in (1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                _size(target_compression_ratio=bad)

    def test_nonpositive_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            _size(token_budget=0)

    def test_fixed_mode_overshoots_the_character_floor(self):
        sizing = _size(
            target_compression_ratio=0.5,
            chars_per_token=3.8,
            char_overshoot=FIXED_CHUNK_CHAR_OVERSHOOT,
        )
        self.assertEqual(sizing.chars, 59266)
        self.assertEqual(sizing.char_overshoot, 1.15)

    def test_fixed_character_cap_cannot_squeeze_the_floor(self):
        floor = 65_536
        cap = fixed_chunk_char_cap(floor)
        self.assertGreaterEqual(cap - cap // 4, floor + 4000)


class ClampTest(unittest.TestCase):
    """Each cap must be able to bind, and must say so."""

    def test_cli_cap_binds_and_reserves_room_for_question_and_answer(self):
        sizing = _size(cli_max_context_tokens=34_000, reserve_tokens=1024)
        self.assertEqual(sizing.tokens, 34_000 - 1024)
        self.assertEqual(sizing.binding, "cli_cap")

    def test_sub_window_binds(self):
        sizing = _size(sub_window_tokens=8192, reserve_tokens=1024)
        self.assertEqual(sizing.tokens, 8192 - 1024)
        self.assertEqual(sizing.binding, "sub_window")

    def test_gpu_fit_binds_on_a_nearly_full_device(self):
        sizing = _size(gpu_free_bytes=3 * 1024**3)
        self.assertEqual(sizing.binding, "gpu_fit")
        self.assertEqual(sizing.tokens, gpu_fit_token_cap(3 * 1024**3, QWEN3_4B_KV_BYTES_PER_TOKEN))

    def test_the_smallest_cap_wins_and_every_candidate_is_recorded(self):
        sizing = _size(sub_window_tokens=262_144, cli_max_context_tokens=34_000, gpu_free_bytes=40 * 1024**3)
        self.assertEqual(sizing.binding, "cli_cap")
        self.assertEqual(set(sizing.caps), {"budget", "sub_window", "cli_cap", "gpu_fit"})
        self.assertEqual(min(v for v in sizing.caps.values() if v is not None), sizing.tokens)

    def test_a_pathologically_tight_cap_refuses_to_size_the_run(self):
        """A GPU that provably cannot hold even min_tokens must stop the run up
        front, not advertise a floor-sized chunk that every sub-call then fails."""
        with self.assertRaises(RuntimeError):
            _size(gpu_free_bytes=KV_FIT_HEADROOM_BYTES, min_tokens=1024)

    def test_uncompressed_tokens_records_the_pre_clamp_ask(self):
        sizing = _size(target_compression_ratio=0.9, cli_max_context_tokens=34_000)
        self.assertEqual(sizing.uncompressed_tokens, ONE_GB_TOKEN_BUDGET * 10)
        self.assertLess(sizing.tokens, sizing.uncompressed_tokens)

    def test_strict_grid_cell_raises_instead_of_accepting_a_binding_cap(self):
        with self.assertRaisesRegex(RuntimeError, "different grid cell"):
            _size(
                target_compression_ratio=0.9,
                cli_max_context_tokens=34_000,
                require_budget_binding=True,
            )


class GpuFitTest(unittest.TestCase):
    def test_it_inverts_the_runtime_fit_check_exactly(self):
        """Anti-drift: gpu_fit_token_cap must be the exact inverse of
        kvzip_backend._fits_in_memory's `tokens * bytes * 1.2 + 1 GiB <= free`.
        If either side is edited alone, this fails."""

        def fits(tokens, free_bytes):
            return tokens * QWEN3_4B_KV_BYTES_PER_TOKEN * KV_FIT_SAFETY_FACTOR + KV_FIT_HEADROOM_BYTES <= free_bytes

        for free_gib in (2, 8, 20, 47):
            free_bytes = free_gib * 1024**3
            cap = gpu_fit_token_cap(free_bytes, QWEN3_4B_KV_BYTES_PER_TOKEN)
            self.assertTrue(fits(cap, free_bytes), f"cap {cap} should fit in {free_gib} GiB")
            self.assertFalse(fits(cap + 1, free_bytes), f"cap {cap} should be maximal in {free_gib} GiB")

    def test_a_device_with_less_free_than_the_headroom_fits_nothing(self):
        self.assertEqual(gpu_fit_token_cap(KV_FIT_HEADROOM_BYTES // 2, QWEN3_4B_KV_BYTES_PER_TOKEN), 0)

    def test_the_reference_number(self):
        self.assertEqual(gpu_fit_token_cap(20 * 1024**3, QWEN3_4B_KV_BYTES_PER_TOKEN), 115_294)


class CalibrationTest(unittest.TestCase):
    def test_it_measures_the_real_ratio(self):
        # 4 chars per "token" exactly.
        sample = "abcd" * 5000
        value, source = calibrate_chars_per_token(sample, lambda s: range(len(s) // 4))
        self.assertAlmostEqual(value, 4.0)
        self.assertEqual(source, "calibrated")

    def test_it_samples_the_middle_not_the_boilerplate_at_the_front(self):
        """LOFT and RULER both open with headers that tokenize unrepresentatively."""
        head = "H" * 1000
        body = "B" * 10_000
        seen = {}

        def encode(s):
            seen["text"] = s
            return range(len(s) // 4)

        calibrate_chars_per_token(head + body, encode, sample_chars=1000)
        self.assertNotIn("H", seen["text"])

    def test_an_implausible_measurement_is_clamped_and_labelled(self):
        low, high = CHARS_PER_TOKEN_BOUNDS
        value, source = calibrate_chars_per_token("x" * 100, lambda s: range(1))
        self.assertEqual((value, source), (high, "clamped"))

        value, source = calibrate_chars_per_token("x" * 100, lambda s: range(1000))
        self.assertEqual((value, source), (low, "clamped"))

    def test_a_tokenizer_failure_falls_back_rather_than_killing_the_run(self):
        def boom(_):
            raise RuntimeError("tokenizer exploded")

        value, source = calibrate_chars_per_token("some text", boom)
        self.assertEqual((value, source), (DEFAULT_CHARS_PER_TOKEN, "fallback"))

    def test_empty_text_falls_back(self):
        value, source = calibrate_chars_per_token("", lambda s: range(10))
        self.assertEqual((value, source), (DEFAULT_CHARS_PER_TOKEN, "fallback"))

    def test_the_calibrated_ratio_is_what_converts_tokens_to_chars(self):
        sizing = _size(chars_per_token=2.5, chars_per_token_source="calibrated")
        self.assertEqual(sizing.chars, int(sizing.tokens * 2.5))
        self.assertEqual(sizing.chars_per_token_source, "calibrated")


class AgreementWithKvpressTest(unittest.TestCase):
    """The whole design rests on inverting kvpress's own formula, so check against it."""

    def test_realized_ratio_matches_the_pipeline_staticmethod(self):
        from kvpress.pipeline import KVPressTextGenerationPipeline

        for target in (0.0, 0.5, 0.75, 0.9):
            sizing = _size(target_compression_ratio=target)
            _, ratio = KVPressTextGenerationPipeline._compute_context_compression_ratio(
                sizing.tokens, sizing.token_budget
            )
            self.assertAlmostEqual(sizing.realized_ratio_if_filled, ratio, places=9)


if __name__ == "__main__":
    unittest.main()
