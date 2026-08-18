"""Download or offline-validate the Synthetic-KV 32K dataset."""

import argparse
import os

from datasets import DownloadConfig, load_dataset


DATASET_ID = "ollamaweights/synthetic-dataset-1208"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force-redownload", action="store_true")
    mode.add_argument("--offline-check", action="store_true")
    args = parser.parse_args()

    if args.offline_check:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    dataset = load_dataset(
        DATASET_ID,
        split="test",
        download_mode=(
            "force_redownload"
            if args.force_redownload
            else "reuse_dataset_if_exists"
        ),
        download_config=DownloadConfig(local_files_only=args.offline_check),
    )
    row = dataset[0]
    required = {
        "context_id",
        "context",
        "questions",
        "answers",
        "answer_prefix",
        "context_tokens",
        "max_new_tokens",
    }
    missing = required.difference(dataset.column_names)
    if len(dataset) != 1 or missing or len(row["questions"]) != len(row["answers"]):
        raise RuntimeError(
            f"Invalid Synthetic-KV 32K dataset; missing columns: {sorted(missing)}"
        )
    print(
        f"Ready {DATASET_ID}: contexts=1, questions={len(row['questions'])}, "
        f"context_tokens={row['context_tokens']}"
    )


if __name__ == "__main__":
    main()
