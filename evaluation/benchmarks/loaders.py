# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Centralized Hugging Face loading and normalization for evaluation datasets.

The returned DataFrame is the source of truth for both KVPress and RLM.  The
``iter_benchmark_examples`` adapter exposes those same rows to orchestration
backends that do not use pandas during inference.
"""

import logging
from typing import Any, Iterator, Mapping, Optional

import pandas as pd
from datasets import load_dataset

from evaluation.benchmarks.registry import RULER_32K_TASKS

logger = logging.getLogger(__name__)


def _require_task(dataset_name: str, task: Optional[str]) -> str:
    if task is None:
        raise ValueError(f"{dataset_name} requires a task name in data_dir")
    return task


def _load_loft(task: str) -> pd.DataFrame:
    parts = task.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid LOFT subset {task!r}; expected dataset_length")

    length = parts[-1]
    dataset_name = "_".join(parts[:-1])
    dataset_id = f"f20180301/loft-rag-{dataset_name}-{length}"
    dataset_dict = load_dataset(dataset_id)

    split_frames = []
    for split_name in ("dev", "test"):
        if split_name in dataset_dict:
            split_df = dataset_dict[split_name].to_pandas()
            split_df["split"] = split_name
            split_frames.append(split_df)
    if not split_frames:
        raise ValueError(f"No dev or test split found for {task} ({dataset_id})")

    df = pd.concat(split_frames, ignore_index=True)
    df["task"] = task

    required_columns = [
        "context",
        "question",
        "answers",
        "task",
        "answer_prefix",
        "max_new_tokens",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"  ✓ Loaded {len(df)} LOFT samples from {task}")
    return df


def _load_synthetic_kv(
    dataset_name: str,
    dataset_id: str,
    task: str,
    metadata_override: bool,
) -> pd.DataFrame:
    expected_task = "32k" if dataset_name == "synthetic_kv_32k" else "64k"
    if task != expected_task:
        raise ValueError(f"Unknown synthetic-KV configuration {task!r}; expected {expected_task!r}")
    if metadata_override:
        raise ValueError("Synthetic-KV metadata override is disabled: use the Hugging Face context")

    logger.info("Loading native Synthetic-KV context from %s", dataset_id)
    dataset = load_dataset(dataset_id, split="test")
    expanded_rows: list[dict[str, Any]] = []
    for compact_row in dataset:
        context = compact_row["context"]
        questions = compact_row["questions"]
        answers = compact_row["answers"]
        if len(questions) != len(answers):
            raise ValueError(
                f"Mismatched questions/answers for {compact_row['context_id']}: " f"{len(questions)} != {len(answers)}"
            )

        max_new_tokens = int(compact_row.get("max_new_tokens", 32))
        context_length = int(compact_row.get("context_tokens", 65536))
        for question, answer in zip(questions, answers):
            expanded_rows.append(
                {
                    "context_id": compact_row["context_id"],
                    "context": context,
                    "question": question,
                    "answer": [answer],
                    "task": f"synthetic_kv_{expected_task}",
                    "answer_prefix": str(compact_row.get("answer_prefix", "")),
                    "max_new_tokens": max_new_tokens,
                    "context_length": context_length,
                }
            )
    if not expanded_rows:
        raise ValueError("The synthetic-KV dataset contains no questions")

    print(
        f"  ✓ Expanded {len(dataset)} Synthetic-KV context(s) into "
        f"{len(expanded_rows)} questions || context tokens = "
        f"{expanded_rows[0]['context_length']}"
    )
    return pd.DataFrame(expanded_rows)


def _load_ruler32k(dataset_id: str, task: str) -> pd.DataFrame:
    """Load one task using sparse-attention-hub's RULER32K layout."""
    if task not in RULER_32K_TASKS:
        valid_tasks = ", ".join(RULER_32K_TASKS)
        raise ValueError(f"Unknown RULER32K task {task!r}; expected one of: {valid_tasks}")

    logger.info("Loading RULER32K dataset: %s (config=%s, split=%s)", dataset_id, task, task)
    # Source contract:
    # https://github.com/skylight-org/sparse-attention-hub/tree/main/benchmark/ruler32k
    # Each task is stored under a same-named Hugging Face config and split.
    df = load_dataset(dataset_id, task, split=task).to_pandas()
    df["context_length"] = 32768
    return df


def load_benchmark_dataset(
    dataset_name: str,
    task: Optional[str],
    dataset_registry: Mapping[str, str],
    synthetic_metadata_override: bool = False,
) -> pd.DataFrame:
    """Load one evaluation task into the shared pandas representation."""
    if dataset_name not in dataset_registry:
        raise ValueError(f"Unknown evaluation dataset: {dataset_name!r}")
    dataset_id = dataset_registry[dataset_name]

    if dataset_name == "loft":
        df = _load_loft(_require_task(dataset_name, task))
    elif dataset_name in {"synthetic_kv", "synthetic_kv_32k"}:
        df = _load_synthetic_kv(
            dataset_name,
            dataset_id,
            _require_task(dataset_name, task),
            synthetic_metadata_override,
        )
    elif dataset_name == "ruler32k":
        df = _load_ruler32k(dataset_id, _require_task(dataset_name, task))
    else:
        logger.info("Loading dataset: %s (data_dir: %s)", dataset_id, task)
        df = load_dataset(dataset_id, data_dir=task, split="test").to_pandas()

    return df


def _answers_from_row(row: Mapping[str, Any]) -> list[str]:
    answer = row.get("answer")
    if answer is None:
        answer = row.get("answers")
    if answer is None:
        raise ValueError("Benchmark row has neither 'answer' nor 'answers'")
    if isinstance(answer, (list, tuple)):
        return [str(value) for value in answer]
    if hasattr(answer, "tolist") and not isinstance(answer, str):
        converted = answer.tolist()
        if isinstance(converted, list):
            return [str(value) for value in converted]
    return [str(answer)]


def _scoring_fields_from_row(row: Mapping[str, Any], answers: list[str]) -> dict[str, Any]:
    """Keep the canonical scorer inputs in a JSON-serializable form."""
    raw_answer = row.get("answer")
    if raw_answer is None:
        raw_answer = answers
    elif hasattr(raw_answer, "tolist") and not isinstance(raw_answer, str):
        raw_answer = raw_answer.tolist()

    fields: dict[str, Any] = {"answer": raw_answer, "answers": answers}
    for name in ("all_classes", "difficulty", "length"):
        value = row.get(name)
        if value is None:
            continue
        if hasattr(value, "tolist") and not isinstance(value, str):
            value = value.tolist()
        elif hasattr(value, "item") and not isinstance(value, str):
            value = value.item()
        fields[name] = value
    return fields


def iter_benchmark_examples(
    df: pd.DataFrame,
    dataset_name: str,
    task: Optional[str],
    limit: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    """Adapt shared benchmark rows to the backend-neutral example contract."""
    required = {"context", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required benchmark columns: {sorted(missing)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    rows = df.head(limit) if limit is not None else df
    for position, (_, series) in enumerate(rows.iterrows()):
        row = series.to_dict()
        source_id = row.get("_id") or row.get("id") or row.get("context_id")
        example_id = f"{source_id}-{position}" if source_id is not None else str(position)
        answers = _answers_from_row(row)
        yield {
            "id": f"{dataset_name}:{task or 'default'}:{example_id}",
            "context": str(row["context"]),
            "question": str(row["question"]),
            "answers": answers,
            "task": str(row.get("task") or task or dataset_name),
            "answer_prefix": str(row.get("answer_prefix") or ""),
            "max_new_tokens": row.get("max_new_tokens"),
            "scoring": _scoring_fields_from_row(row, answers),
        }
