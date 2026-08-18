"""Tests for the compact synthetic key-value generator."""

from __future__ import annotations

import random
import unittest

from generate_dataset import (
    CONTEXT_HEADER,
    _format_context,
    _generate_fixed_pairs,
    _generate_to_token_budget,
)


class CharacterTokenizer:
    """Small deterministic tokenizer used to test budget enforcement."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return list(text)


class GenerateDatasetTests(unittest.TestCase):
    def test_context_uses_one_instruction_and_kv_array(self) -> None:
        context = _format_context([("ABC", "123"), ("DEF", "456")])
        self.assertEqual(
            context,
            CONTEXT_HEADER + "[\n[K_ABC: V_123],\n[K_DEF: V_456]\n]",
        )
        self.assertEqual(context.count(CONTEXT_HEADER), 1)
        self.assertNotIn("<record>", context)
        self.assertNotIn("KEY:", context)
        self.assertNotIn("VALUE:", context)

    def test_fixed_pairs_are_reproducible_and_unique(self) -> None:
        first = _generate_fixed_pairs(random.Random(42), 100, 12, 12)
        second = _generate_fixed_pairs(random.Random(42), 100, 12, 12)
        self.assertEqual(first, second)
        self.assertEqual(len({key for key, _ in first}), 100)
        self.assertEqual(len({value for _, value in first}), 100)

    def test_token_budget_is_never_exceeded(self) -> None:
        pairs, token_count = _generate_to_token_budget(
            rng=random.Random(42),
            tokenizer=CharacterTokenizer(),
            target_context_tokens=len(CONTEXT_HEADER) + 50,
            key_hex_length=4,
            value_hex_length=4,
        )
        self.assertTrue(pairs)
        self.assertLessEqual(token_count, len(CONTEXT_HEADER) + 50)
        next_entry_cost = len(",\n[K_0000: V_0000]")
        self.assertGreater(token_count + next_entry_cost, len(CONTEXT_HEADER) + 50)


if __name__ == "__main__":
    unittest.main()

