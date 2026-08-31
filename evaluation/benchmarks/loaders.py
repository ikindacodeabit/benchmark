# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Centralized Hugging Face loading and normalization for evaluation datasets.

The returned DataFrame is the source of truth for both KVPress and RLM.  The
``iter_benchmark_examples`` adapter exposes those same rows to orchestration
backends that do not use pandas during inference.
"""

import logging
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import pandas as pd
from datasets import load_dataset

from evaluation.benchmarks.registry import LONGBENCH_128K_SUBSETS, RULER_32K_TASKS

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


def _load_longbench128k(task: str) -> pd.DataFrame:
    """Load one locally built, uniformly padded LongBench-128K subset."""
    if task not in LONGBENCH_128K_SUBSETS:
        valid = ", ".join(LONGBENCH_128K_SUBSETS)
        raise ValueError(f"Unknown LongBench-128K subset {task!r}; expected one of: {valid}")

    data_root = Path(os.environ.get("RLM_DATA_DIR", os.path.expanduser("~/rlm_data")))
    parquet_path = data_root / "longbench128k" / task / "data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} is missing; build the local pack with "
            "python -m evaluation.benchmarks.longbench128k.build_dataset"
        )
    df = pd.read_parquet(parquet_path)
    required = {"context", "question", "answers", "all_classes", "task", "gold_depth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LongBench-128K parquet is missing columns: {sorted(missing)}")
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


def _load_ruler64k(dataset_id: str, task: str) -> pd.DataFrame:
    """Load one task from the Qwen-tokenized RULER64K dataset."""
    logger.info("Loading RULER64K dataset: %s (config=65536, task=%s)", dataset_id, task)
    dataset = load_dataset(dataset_id, "65536", split="test")

    available_tasks = set(dataset.unique("task"))
    if task not in available_tasks:
        valid_tasks = ", ".join(sorted(available_tasks))
        raise ValueError(f"Unknown RULER64K task {task!r}; expected one of: {valid_tasks}")

    df = dataset.filter(lambda example: example["task"] == task).to_pandas()
    df["context_length"] = 65536
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
    elif dataset_name == "longbench128k":
        df = _load_longbench128k(_require_task(dataset_name, task))
    elif dataset_name in {"synthetic_kv", "synthetic_kv_32k"}:
        df = _load_synthetic_kv(
            dataset_name,
            dataset_id,
            _require_task(dataset_name, task),
            synthetic_metadata_override,
        )
    elif dataset_name == "ruler32k":
        df = _load_ruler32k(dataset_id, _require_task(dataset_name, task))
    elif dataset_name == "ruler64k":
        df = _load_ruler64k(dataset_id, _require_task(dataset_name, task))
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
    for name in ("all_classes", "difficulty", "length", "gold_depth"):
        value = row.get(name)
        if value is None:
            # LongBench's QA rows explicitly carry all_classes=null. Its scorer
            # still indexes that column before dispatching to qa_f1_score, so
            # dropping the field turns a valid QA run into a KeyError at final
            # scoring time. Preserve an explicit null; continue ignoring fields
            # that are genuinely absent from the source schema.
            if name == "all_classes" and name in row:
                fields[name] = None
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
    split: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    """Adapt shared benchmark rows to the backend-neutral example contract."""
    required = {"context", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required benchmark columns: {sorted(missing)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    if split and split != "all" and "split" in df.columns:
        df = df[df["split"] == split]
        if df.empty:
            raise ValueError(f"{dataset_name}/{task or 'default'} has no rows in split {split!r}")

    rows = df.head(limit) if limit is not None else df
    # LOFT concatenates dev (10 rows) then test (100), so `--limit 10` with no split
    # filter silently evaluates the dev split alone -- a smoke run that lands in the
    # results tree indistinguishable from a real one. Say so rather than let a 0.000
    # be read as a result.
    if limit is not None and "split" in rows.columns:
        covered = set(rows["split"].dropna().unique())
        available = set(df["split"].dropna().unique())
        if len(available) > 1 and len(covered) == 1:
            only = covered.pop()
            logger.warning(
                "--limit %d selects only the %r split of %s/%s (%d of %d rows). "
                "Pass --split test for a real run, or --split %s to make this explicit.",
                limit,
                only,
                dataset_name,
                task or "default",
                len(rows),
                len(df),
                only,
            )
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
            "split": row.get("split"),
            "scoring": _scoring_fields_from_row(row, answers),
        }
