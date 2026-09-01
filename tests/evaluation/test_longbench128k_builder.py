# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the local LongBench-128K pack builder."""

import pandas as pd

from evaluation.benchmarks.loaders import iter_benchmark_examples
from evaluation.benchmarks.longbench128k.build_dataset import SourceExample, build_subset
from evaluation.benchmarks.longbench128k.calculate_metrics import calculate_metrics


class CharTokenizer:
    """One Unicode codepoint per token, with fast-tokenizer-style offsets."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _examples():
    return [
        SourceExample(
            source_id=f"id-{index}",
            subset="hotpotqa",
            question=f"question {index}",
            context=f"<gold-{index}>" + chr(65 + index) * 45,
            answers=[f"gold-{index}"],
            all_classes=None,
        )
        for index in range(8)
    ]


def test_every_pack_is_exact_and_preserves_the_gold_once():
    rows, manifest = build_subset(_examples(), CharTokenizer(), per_subset=4, target_tokens=180, seed=0)

    assert len(rows) == 4
    for row, source in zip(rows, _examples()):
        assert len(row["context"]) == 180
        assert row["context"].count(source.context) == 1
        assert row["context_length"] == 180
        assert 0.0 <= row["gold_depth"] < 1.0
    assert manifest["packed_answer_in_context_rate"] >= manifest["native_answer_in_context_rate"]


def test_build_is_deterministic_and_pilot_rows_match_full_rows():
    examples = _examples()
    pilot, pilot_manifest = build_subset(examples, CharTokenizer(), per_subset=2, target_tokens=180, seed=7)
    full, full_manifest = build_subset(examples, CharTokenizer(), per_subset=4, target_tokens=180, seed=7)

    assert pilot == full[:2]
    assert pilot_manifest["examples"] == full_manifest["examples"][:2]


def test_seed_changes_distractor_layout_without_changing_the_gold():
    examples = _examples()
    first, _ = build_subset(examples, CharTokenizer(), per_subset=1, target_tokens=180, seed=1)
    second, _ = build_subset(examples, CharTokenizer(), per_subset=1, target_tokens=180, seed=2)

    assert first[0]["context"] != second[0]["context"]
    assert examples[0].context in first[0]["context"]
    assert examples[0].context in second[0]["context"]


def test_null_all_classes_survives_the_loader_scorer_boundary():
    rows, _ = build_subset(_examples(), CharTokenizer(), per_subset=1, target_tokens=180, seed=0)
    example = next(iter_benchmark_examples(pd.DataFrame(rows), "longbench128k", "hotpotqa"))

    assert "all_classes" in example["scoring"]
    assert example["scoring"]["all_classes"] is None
    frame = pd.DataFrame(
        [
            {
                **example["scoring"],
                "predicted_answer": example["answers"][0],
                "task": example["task"],
            }
        ]
    )
    assert calculate_metrics(frame) == 100.0
