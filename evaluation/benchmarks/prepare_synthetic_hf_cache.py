# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download or offline-validate the Synthetic-KV benchmark datasets."""

import argparse
import os
from pathlib import Path
from typing import Any


DATASETS = {
    "32k": "ollamaweights/synthetic-dataset-1208",
    "64k": "ollamaweights/synthetic-dataset-1208-64k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["32k", "64k", "all"],
        default="all",
        help="Dataset variant to prepare (default: all).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force-redownload",
        action="store_true",
        help="Fetch a fresh copy from Hugging Face instead of reusing the cache.",
    )
    mode.add_argument(
        "--offline-check",
        action="store_true",
        help="Require the dataset to be available locally and make no network request.",
    )
    return parser.parse_args()


def validate_dataset(dataset: Any, dataset_id: str, variant: str) -> None:
    """Validate the compact Synthetic-KV schema required by evaluation."""
    if len(dataset) != 1:
        raise RuntimeError(f"Expected one compact context, found {len(dataset)}")

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
        raise RuntimeError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    row = dataset[0]
    if len(row["questions"]) != len(row["answers"]):
        raise RuntimeError("Dataset questions and answers have different lengths")
    if not row["questions"]:
        raise RuntimeError("Dataset contains no question/answer pairs")
    context = row["context"]
    if not isinstance(context, str) or not context.strip():
        raise RuntimeError("Dataset context must be a non-empty string")

    print(
        f"Ready {dataset_id}/{variant}: contexts={len(dataset)}, "
        f"questions={len(row['questions'])}, context_tokens={row['context_tokens']}, "
        f"max_new_tokens={row['max_new_tokens']}, "
        f"answer_prefix={row['answer_prefix']!r}"
    )


def main() -> None:
    args = parse_args()
    if args.offline_check:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from datasets import DownloadConfig, load_dataset

    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    datasets_cache = Path(
        os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets")
    ).expanduser()
    mode = "offline check" if args.offline_check else "online/cache preparation"
    print(f"Mode: {mode}")
    print(f"Hugging Face datasets cache: {datasets_cache}")

    variants = DATASETS if args.variant == "all" else {args.variant: DATASETS[args.variant]}
    for variant, dataset_id in variants.items():
        dataset = load_dataset(
            dataset_id,
            split="test",
            download_mode=(
                "force_redownload" if args.force_redownload else "reuse_dataset_if_exists"
            ),
            download_config=DownloadConfig(local_files_only=args.offline_check),
        )
        validate_dataset(dataset, dataset_id, variant)


if __name__ == "__main__":
    main()
