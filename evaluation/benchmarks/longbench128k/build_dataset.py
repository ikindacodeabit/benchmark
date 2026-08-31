# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a uniform 128K-token QA pack from the local LongBench JSONL.

The native LongBench contexts are preserved verbatim and padded with contexts
from other examples in the same subset. Only the final distractor may be
trimmed. The output is local by design: one parquet and one audit manifest per
subset under ``$RLM_DATA_DIR/longbench128k``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from evaluation.benchmarks.registry import LONGBENCH_128K_SUBSETS

DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TARGET_TOKENS = 131_072
DEFAULT_PER_SUBSET = 60
SEPARATOR = "\n\n"
MAX_NEW_TOKENS = {"narrativeqa": 128, "qasper": 128}


@dataclass(frozen=True)
class SourceExample:
    source_id: str
    subset: str
    question: str
    context: str
    answers: list[str]
    all_classes: Any


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _answer_in_context(answers: Sequence[str], context: str) -> bool:
    folded = context.casefold()
    return any(str(answer).casefold() in folded for answer in answers if str(answer))


def _trim_tail_to_exact_tokens(tokenizer: Any, prefix: str, tail: str, target_tokens: int) -> str:
    """Trim only ``tail`` so ``prefix + tail`` tokenizes to exactly the target."""
    full = prefix + tail
    full_ids = _encode(tokenizer, full)
    if len(full_ids) < target_tokens:
        raise ValueError(f"padding pool supplied only {len(full_ids)} of {target_tokens} required tokens")
    if len(full_ids) == target_tokens:
        return full

    # Fast tokenizers expose exact character offsets. Slicing at the target
    # token boundary preserves every preceding character, including the gold.
    try:
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        char_end = int(offsets[target_tokens - 1][1])
        candidate = full[:char_end]
        if char_end >= len(prefix) and len(_encode(tokenizer, candidate)) == target_tokens:
            return candidate
    except (KeyError, TypeError, ValueError, NotImplementedError):
        pass

    # A slow tokenizer has no offsets. Decoding the target prefix is cheap and
    # safe when it reproduces the original, exact prefix; otherwise use a
    # bounded binary search over the final distractor's characters.
    try:
        decoded = tokenizer.decode(
            full_ids[:target_tokens], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded.startswith(prefix) and len(_encode(tokenizer, decoded)) == target_tokens:
            return decoded
    except (AttributeError, TypeError, ValueError):
        pass

    if len(_encode(tokenizer, prefix)) >= target_tokens:
        raise ValueError("the untrimmed prefix already reaches the target; trimming only the tail is impossible")
    low, high = 0, len(tail)
    while low < high:
        middle = (low + high + 1) // 2
        if len(_encode(tokenizer, prefix + tail[:middle])) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    for end in range(max(0, low - 64), min(len(tail), low + 64) + 1):
        candidate = prefix + tail[:end]
        if len(_encode(tokenizer, candidate)) == target_tokens:
            return candidate
    raise ValueError("could not trim the final distractor to an exact token boundary")


def load_source(path: Path, subsets: Iterable[str] = LONGBENCH_128K_SUBSETS) -> dict[str, list[SourceExample]]:
    wanted = set(subsets)
    rows: dict[str, list[SourceExample]] = {subset: [] for subset in wanted}
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            subset = raw.get("dataset")
            if subset not in wanted:
                continue
            try:
                rows[subset].append(
                    SourceExample(
                        source_id=str(raw["_id"]),
                        subset=subset,
                        question=str(raw["input"]),
                        context=str(raw["context"]),
                        answers=[str(answer) for answer in raw["answers"]],
                        all_classes=raw.get("all_classes"),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} is missing {exc.args[0]!r}") from exc
    missing = [subset for subset, examples in rows.items() if not examples]
    if missing:
        raise ValueError(f"{path} contains no rows for: {', '.join(sorted(missing))}")
    return rows


def build_subset(
    examples: Sequence[SourceExample],
    tokenizer: Any,
    *,
    per_subset: int = DEFAULT_PER_SUBSET,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one subset and return parquet rows plus its manifest."""
    if not examples:
        raise ValueError("examples must not be empty")
    if per_subset < 1 or per_subset > len(examples):
        raise ValueError(f"per_subset must be in [1, {len(examples)}], got {per_subset}")
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")

    subset = examples[0].subset
    if any(example.subset != subset for example in examples):
        raise ValueError("build_subset received mixed subsets")

    token_lengths = {example.source_id: len(_encode(tokenizer, example.context)) for example in examples}
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    native_answer_hits = 0
    packed_answer_hits = 0

    for index, gold in enumerate(examples[:per_subset]):
        gold_tokens = token_lengths[gold.source_id]
        if gold_tokens >= target_tokens:
            raise ValueError(
                f"gold {gold.source_id} is {gold_tokens} tokens, so it cannot be preserved inside "
                f"a {target_tokens}-token pack"
            )

        # Stable per example: changing --per-subset does not change the first N
        # packs, which keeps the pilot rows identical to the full campaign rows.
        rng = random.Random(f"{seed}:{subset}:{gold.source_id}")
        candidates = [
            example
            for example in examples
            if example.source_id != gold.source_id and gold.context not in example.context
        ]
        rng.shuffle(candidates)
        unique_candidates = []
        seen_contexts = set()
        for candidate in candidates:
            if candidate.context in seen_contexts:
                continue
            seen_contexts.add(candidate.context)
            unique_candidates.append(candidate)
        candidates = unique_candidates

        selected: list[SourceExample] = []
        selected_tokens = 0
        needed = target_tokens - gold_tokens
        for distractor in candidates:
            selected.append(distractor)
            selected_tokens += token_lengths[distractor.source_id]
            if selected_tokens >= needed:
                break
        if selected_tokens < needed:
            raise ValueError(f"{subset} does not have enough distinct distractor text for {gold.source_id}")

        # The threshold-crossing distractor is the final, trimmable one. This is
        # the literal seeded draw order; if separator tokens make that prefix too
        # large, reserve the largest distractor instead as a safety fallback.
        tail = selected[-1]
        body = selected[:-1]
        split_fraction = rng.random()
        split_at = round(split_fraction * len(body))
        before, after = body[:split_at], body[split_at:]
        prefix_segments = (
            [example.context for example in before] + [gold.context] + [example.context for example in after]
        )
        prefix = SEPARATOR.join(prefix_segments) + SEPARATOR
        if len(_encode(tokenizer, prefix)) >= target_tokens:
            tail = max(selected, key=lambda example: token_lengths[example.source_id])
            body = [example for example in selected if example.source_id != tail.source_id]
            split_at = round(split_fraction * len(body))
            before, after = body[:split_at], body[split_at:]
            prefix = (
                SEPARATOR.join(
                    [example.context for example in before] + [gold.context] + [example.context for example in after]
                )
                + SEPARATOR
            )
        packed = _trim_tail_to_exact_tokens(tokenizer, prefix, tail.context, target_tokens)

        actual_tokens = len(_encode(tokenizer, packed))
        if actual_tokens != target_tokens:
            raise AssertionError(f"{gold.source_id}: built {actual_tokens}, expected {target_tokens} tokens")
        if packed.count(gold.context) != 1:
            raise AssertionError(f"{gold.source_id}: gold context must appear verbatim exactly once")

        gold_char_start = packed.index(gold.context)
        gold_depth = len(_encode(tokenizer, packed[:gold_char_start])) / target_tokens
        native_hit = _answer_in_context(gold.answers, gold.context)
        packed_hit = _answer_in_context(gold.answers, packed)
        native_answer_hits += int(native_hit)
        packed_answer_hits += int(packed_hit)

        distractor_ids = [example.source_id for example in before + after + [tail]]
        rows.append(
            {
                "_id": gold.source_id,
                "context": packed,
                "question": gold.question,
                "answers": gold.answers,
                "all_classes": gold.all_classes,
                "task": subset,
                "max_new_tokens": MAX_NEW_TOKENS.get(subset, 32),
                "answer_prefix": "",
                "gold_depth": gold_depth,
                "context_length": target_tokens,
            }
        )
        audit_rows.append(
            {
                "source_id": gold.source_id,
                "gold_depth": gold_depth,
                "requested_depth_fraction": split_fraction,
                "gold_tokens": gold_tokens,
                "distractor_ids": distractor_ids,
                "context_tokens": actual_tokens,
                "native_answer_in_context": native_hit,
                "packed_answer_in_context": packed_hit,
            }
        )

    native_rate = native_answer_hits / per_subset
    packed_rate = packed_answer_hits / per_subset
    if packed_rate < native_rate:
        raise AssertionError(f"{subset}: answer-in-context rate fell from {native_rate:.3f} to {packed_rate:.3f}")
    manifest = {
        "subset": subset,
        "seed": seed,
        "target_tokens": target_tokens,
        "examples": audit_rows,
        "native_answer_in_context_rate": native_rate,
        "packed_answer_in_context_rate": packed_rate,
    }
    return rows, manifest


def write_subset(rows: list[dict[str, Any]], manifest: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_dir / "data.parquet", index=False)
    (output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    data_root = Path(os.environ.get("RLM_DATA_DIR", os.path.expanduser("~/rlm_data")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data_root / "longbench.jsonl")
    parser.add_argument("--output", type=Path, default=data_root / "longbench128k")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--per-subset", type=int, default=DEFAULT_PER_SUBSET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsets", nargs="+", choices=LONGBENCH_128K_SUBSETS, default=LONGBENCH_128K_SUBSETS)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    by_subset = load_source(args.source, args.subsets)
    for subset in args.subsets:
        rows, manifest = build_subset(
            by_subset[subset],
            tokenizer,
            per_subset=args.per_subset,
            target_tokens=args.target_tokens,
            seed=args.seed,
        )
        manifest["tokenizer_id"] = args.tokenizer
        write_subset(rows, manifest, args.output / subset)
        print(f"{subset}: wrote {len(rows)} rows at exactly {args.target_tokens} tokens to {args.output / subset}")


if __name__ == "__main__":
    main()
