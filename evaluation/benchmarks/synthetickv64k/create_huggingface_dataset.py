# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate and optionally publish the formatted Synthetic-KV dataset."""

import argparse
import random
import re
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_REPO_ID = "ollamaweights/synthetickv_formated"
PUBLISHED_NUM_PAIRS = 1_707
PUBLISHED_CONTEXT_TOKENS = 62_982

CONTEXT_PREFIX = """You are given a collection of key-value records.

Each record contains:
- a key beginning with K_
- a value beginning with V_

Find the record whose key exactly matches the query key.
Return only the complete corresponding value exactly as written.
Do not return the key, an explanation, quotes, punctuation, or any additional text.

Records:
"""
RECORD_TEMPLATE = "<record>\nKEY: {key}\nVALUE: {value}\n</record>"
QUESTION_TEMPLATE = "Query KEY: {key}\nReturn only its VALUE."
ANSWER_PREFIX = "Answer: "
RECORD_PATTERN = re.compile(
    r"<record>\nKEY: (K_[0-9A-F]+)\nVALUE: (V_[0-9A-F]+)\n</record>"
)


def generate_unique_pairs(
    num_pairs: int,
    seed: int,
    identifier_width: int = 12,
) -> list[tuple[str, str]]:
    """Generate globally unique, deterministic K_/V_ hexadecimal identifiers."""
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if identifier_width <= 0:
        raise ValueError("identifier_width must be positive")

    identifier_space = 16**identifier_width
    if 2 * num_pairs > identifier_space:
        raise ValueError(
            "identifier_width is too small for globally unique keys and values"
        )

    generator = random.Random(seed)
    used_identifiers: set[str] = set()

    def next_identifier(prefix: str) -> str:
        while True:
            identifier = f"{generator.randrange(identifier_space):0{identifier_width}X}"
            if identifier not in used_identifiers:
                used_identifiers.add(identifier)
                return f"{prefix}_{identifier}"

    return [(next_identifier("K"), next_identifier("V")) for _ in range(num_pairs)]


def format_context(pairs: list[tuple[str, str]]) -> str:
    """Build the exact instruction and XML-like record format used on Hugging Face."""
    records = "\n".join(
        RECORD_TEMPLATE.format(key=key, value=value) for key, value in pairs
    )
    return f"{CONTEXT_PREFIX}{records}\n"


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count context tokens without model special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def select_pairs_for_token_target(
    candidates: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    target_context_tokens: int,
) -> tuple[list[tuple[str, str]], int]:
    """Select the largest candidate prefix that does not exceed the token target."""
    if target_context_tokens <= 0:
        raise ValueError("target_context_tokens must be positive")
    if not candidates:
        raise ValueError("At least one candidate pair is required")

    low = 1
    high = len(candidates)
    best_count = 0
    best_tokens = 0
    while low <= high:
        midpoint = (low + high) // 2
        context_tokens = count_tokens(tokenizer, format_context(candidates[:midpoint]))
        if context_tokens <= target_context_tokens:
            best_count = midpoint
            best_tokens = context_tokens
            low = midpoint + 1
        else:
            high = midpoint - 1

    if best_count == 0:
        raise ValueError(
            "The context instructions and first record exceed the token target"
        )
    if best_count == len(candidates):
        raise ValueError(
            "All candidate pairs fit; increase --candidate-pairs to fill the token target"
        )
    return candidates[:best_count], best_tokens


def build_compact_dataset(
    pairs: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    context_id: str = "context_0000",
    max_new_tokens: int = 32,
) -> Dataset:
    """Store one shared context with aligned question and answer arrays."""
    context = format_context(pairs)
    return Dataset.from_dict(
        {
            "context_id": [context_id],
            "context": [context],
            "questions": [[QUESTION_TEMPLATE.format(key=key) for key, _ in pairs]],
            "answers": [[value for _, value in pairs]],
            "answer_prefix": [ANSWER_PREFIX],
            "context_tokens": [count_tokens(tokenizer, context)],
            "max_new_tokens": [max_new_tokens],
        }
    )


def validate_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> None:
    """Fail before saving or pushing if schema, formatting, or alignment changed."""
    if len(dataset) != 1:
        raise ValueError(f"Expected one compact context, found {len(dataset)}")

    row = dataset[0]
    required_columns = {
        "context_id",
        "context",
        "questions",
        "answers",
        "answer_prefix",
        "context_tokens",
        "max_new_tokens",
    }
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    records = RECORD_PATTERN.findall(row["context"])
    keys = [key for key, _ in records]
    values = [value for _, value in records]
    expected_questions = [QUESTION_TEMPLATE.format(key=key) for key in keys]
    if not records or row["context"] != format_context(records):
        raise ValueError(
            "Context does not match the published instruction/record format"
        )
    if row["questions"] != expected_questions or row["answers"] != values:
        raise ValueError("Context records are not aligned with questions and answers")
    if len(keys) != len(set(keys)) or len(values) != len(set(values)):
        raise ValueError("Keys and values must each be unique")
    if {key.removeprefix("K_") for key in keys}.intersection(
        value.removeprefix("V_") for value in values
    ):
        raise ValueError("The hexadecimal identifiers must be globally unique")
    if row["answer_prefix"] != ANSWER_PREFIX:
        raise ValueError(f"answer_prefix must be {ANSWER_PREFIX!r}")

    actual_tokens = count_tokens(tokenizer, row["context"])
    if row["context_tokens"] != actual_tokens:
        raise ValueError(
            f"Stored context_tokens={row['context_tokens']} but counted {actual_tokens}"
        )


def verify_published_signature(dataset: Dataset) -> None:
    """Verify the row matches the size of the currently published 64K dataset."""
    row = dataset[0]
    if len(row["questions"]) != PUBLISHED_NUM_PAIRS:
        raise ValueError(
            f"Expected {PUBLISHED_NUM_PAIRS} published pairs, found {len(row['questions'])}"
        )
    if row["context_tokens"] != PUBLISHED_CONTEXT_TOKENS:
        raise ValueError(
            f"Expected {PUBLISHED_CONTEXT_TOKENS} published context tokens, "
            f"found {row['context_tokens']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir", type=Path, help="Optional Dataset.save_to_disk destination"
    )
    parser.add_argument(
        "--push", action="store_true", help="Push the validated test split to --repo-id"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update a private dataset repository",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--target-context-tokens", type=int, default=63_000)
    parser.add_argument("--candidate-pairs", type=int, default=5_000)
    parser.add_argument(
        "--num-pairs",
        type=int,
        help="Use an exact pair count instead of filling the token target",
    )
    parser.add_argument("--identifier-width", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the tokenizer only from the local Hugging Face cache",
    )
    parser.add_argument(
        "--verify-published",
        action="store_true",
        help="Require 1,707 pairs and 62,982 tokens before saving or pushing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_pairs is not None and args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if args.candidate_pairs <= 0:
        raise ValueError("--candidate-pairs must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    candidate_count = args.num_pairs or args.candidate_pairs
    pairs = generate_unique_pairs(candidate_count, args.seed, args.identifier_width)
    selected_tokens: Optional[int] = None
    if args.num_pairs is None:
        pairs, selected_tokens = select_pairs_for_token_target(
            pairs,
            tokenizer,
            args.target_context_tokens,
        )

    dataset = build_compact_dataset(
        pairs, tokenizer, max_new_tokens=args.max_new_tokens
    )
    validate_dataset(dataset, tokenizer)
    if selected_tokens is not None and dataset[0]["context_tokens"] != selected_tokens:
        raise RuntimeError("Context token count changed while building the dataset")
    if args.verify_published:
        verify_published_signature(dataset)

    if args.output_dir is not None:
        dataset.save_to_disk(str(args.output_dir))
        print(f"Saved local dataset to {args.output_dir}")
    if args.push:
        dataset.push_to_hub(args.repo_id, split="test", private=args.private)
        print(f"Pushed test split to https://huggingface.co/datasets/{args.repo_id}")

    row = dataset[0]
    print(
        f"Created contexts={len(dataset)}, questions={len(row['questions'])}, "
        f"context_tokens={row['context_tokens']}, answer_prefix={row['answer_prefix']!r}"
    )
    if not args.push:
        print("Not pushed. Add --push after inspecting the generated dataset.")


if __name__ == "__main__":
    main()
