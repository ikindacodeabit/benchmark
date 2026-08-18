#!/usr/bin/env python3
"""Generate a compact synthetic key-value retrieval dataset.

Each generated context contains a short instruction followed by an array of N
key-value entries:

    [K_<random-key>: V_<random-value>]

For every key in the context, one question is generated asking for its value.
The compact JSONL stores each context only once; dataset.py expands it into N
evaluation rows when KVPress loads the benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


CONTEXT_HEADER = (
    "You are given an array of key-value entries. Every key begins with K_ and "
    "every value begins with V_. Each entry has the format [key: value]. Given "
    "a query key, find its exact match and return only the corresponding value, "
    "exactly as written.\n"
)
DEFAULT_REPO_ID = "ollamaweights/synthetic-dataset-1208-64k"


def _random_identifier(rng: random.Random, n_hex: int, used: set[str]) -> str:
    while True:
        value = f"{rng.getrandbits(n_hex * 4):0{n_hex}X}"
        if value not in used:
            used.add(value)
            return value


def _new_pair(
    rng: random.Random,
    key_hex_length: int,
    value_hex_length: int,
    used_keys: set[str],
    used_values: set[str],
) -> tuple[str, str]:
    key = _random_identifier(rng, key_hex_length, used_keys)
    value = _random_identifier(rng, value_hex_length, used_values)
    return key, value


def _format_context(pairs: list[tuple[str, str]]) -> str:
    # One compact pair replaces the much larger <record>/KEY:/VALUE: block.
    # K_ and V_ make the two roles explicit, while colons and brackets make the
    # collection visually unambiguous without repeating natural-language labels.
    items = ",\n".join(f"[K_{key}: V_{value}]" for key, value in pairs)
    return CONTEXT_HEADER + "[\n" + items + "\n]"


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _generate_fixed_pairs(
    rng: random.Random,
    num_pairs: int,
    key_hex_length: int,
    value_hex_length: int,
) -> list[tuple[str, str]]:
    used_keys: set[str] = set()
    used_values: set[str] = set()
    return [
        _new_pair(rng, key_hex_length, value_hex_length, used_keys, used_values)
        for _ in range(num_pairs)
    ]


def _generate_to_token_budget(
    rng: random.Random,
    tokenizer: Any,
    target_context_tokens: int,
    key_hex_length: int,
    value_hex_length: int,
) -> tuple[list[tuple[str, str]], int]:
    if target_context_tokens <= 0:
        raise ValueError("target_context_tokens must be positive")

    pairs: list[tuple[str, str]] = []
    used_keys: set[str] = set()
    used_values: set[str] = set()

    while True:
        pair = _new_pair(
            rng,
            key_hex_length,
            value_hex_length,
            used_keys,
            used_values,
        )

        candidate_pairs = pairs + [pair]
        candidate_context = _format_context(candidate_pairs)
        candidate_tokens = _token_count(tokenizer, candidate_context)

        # Stop before exceeding the requested token budget.
        if candidate_tokens > target_context_tokens:
            break

        pairs.append(pair)

    if not pairs:
        raise RuntimeError(
            "The token budget is too small to fit even one key-value pair"
        )

    context = _format_context(pairs)
    actual_tokens = _token_count(tokenizer, context)

    return pairs, actual_tokens

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-contexts", type=int, default=1)
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=None,
        help="Exact number N of key-value pairs per context. Produces N questions per context.",
    )
    parser.add_argument(
        "--target-context-tokens",
        type=int,
        default=None,
        help="Automatically choose N so the context is at most this many tokenizer tokens.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Local model/tokenizer path or Hugging Face ID. Required with --target-context-tokens.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key-hex-length", type=int, default=12)
    parser.add_argument("--value-hex-length", type=int, default=12)
    parser.add_argument(
        "--answer-prefix",
        type=str,
        default="Answer: ",
        help="Generation cue appended before the model response.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--push", action="store_true", help="Upload the generated JSONL as the Hugging Face test split.")
    parser.add_argument("--private", action="store_true", help="Make the uploaded Hugging Face dataset private.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_contexts <= 0:
        raise ValueError("--num-contexts must be positive")
    if args.key_hex_length <= 0 or args.value_hex_length <= 0:
        raise ValueError("hex lengths must be positive")
    if args.num_pairs is None and args.target_context_tokens is None:
        args.num_pairs = 1024
    if args.num_pairs is not None and args.target_context_tokens is not None:
        raise ValueError("Use either --num-pairs or --target-context-tokens, not both")
    if args.num_pairs is not None and args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if args.target_context_tokens is not None and not args.tokenizer:
        raise ValueError("--tokenizer is required with --target-context-tokens")

    tokenizer = None
    if args.target_context_tokens is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "test.jsonl"

    metadata: dict[str, Any] = {
        "format": "synthetic_kv_compact_v4_kv_array",
        "context_record_format": "[K_<key>: V_<value>]",
        "question_format": "Query key: K_{key}",
        "answer_prefix": args.answer_prefix,
        "seed": args.seed,
        "num_contexts": args.num_contexts,
        "requested_num_pairs": args.num_pairs,
        "target_context_tokens": args.target_context_tokens,
        "tokenizer": args.tokenizer,
        "key_hex_length": args.key_hex_length,
        "value_hex_length": args.value_hex_length,
        "max_new_tokens": args.max_new_tokens,
        "contexts": [],
    }

    total_questions = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for context_index in range(args.num_contexts):
            rng = random.Random(args.seed + context_index)

            if tokenizer is not None:
                pairs, context_tokens = _generate_to_token_budget(
                    rng=rng,
                    tokenizer=tokenizer,
                    target_context_tokens=args.target_context_tokens,
                    key_hex_length=args.key_hex_length,
                    value_hex_length=args.value_hex_length,
                )
            else:
                pairs = _generate_fixed_pairs(
                    rng=rng,
                    num_pairs=args.num_pairs,
                    key_hex_length=args.key_hex_length,
                    value_hex_length=args.value_hex_length,
                )
                context_tokens = None

            context = _format_context(pairs)
            # The context contains the output instruction once, so repeating it
            # in every question only increases per-query prefill cost.
            questions = [f"Query key: K_{key}" for key, _ in pairs]
            answers = [f"V_{value}" for _, value in pairs]

            record = {
                "context_id": f"context_{context_index:04d}",
                "context": context,
                "questions": questions,
                "answers": answers,
                "answer_prefix": args.answer_prefix,
                "num_pairs": len(pairs),
                "context_tokens": context_tokens,
                "max_new_tokens": args.max_new_tokens,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_questions += len(pairs)
            metadata["contexts"].append(
                {
                    "context_id": record["context_id"],
                    "num_pairs": len(pairs),
                    "num_questions": len(questions),
                    "context_tokens": context_tokens,
                    "context_characters": len(context),
                }
            )

    metadata["total_questions"] = total_questions
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if args.push:
        from datasets import Dataset

        Dataset.from_json(str(output_file)).push_to_hub(
            args.repo_id, split="test", private=args.private
        )
        print(f"Pushed test split to https://huggingface.co/datasets/{args.repo_id}")

    print(f"Wrote compact dataset: {output_file}")
    print(f"Contexts: {args.num_contexts}")
    print(f"Total questions: {total_questions}")
    for item in metadata["contexts"]:
        token_text = (
            f", context_tokens={item['context_tokens']}"
            if item["context_tokens"] is not None
            else ""
        )
        print(
            f"  {item['context_id']}: N={item['num_pairs']} pairs/questions"
            f"{token_text}, characters={item['context_characters']}"
        )


if __name__ == "__main__":
    main()
