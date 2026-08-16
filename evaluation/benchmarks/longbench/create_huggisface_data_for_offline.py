# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from datasets import load_dataset

longbench_tasks = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
    "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news",
    "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count",
    "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
]

base_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/Xnhyacinth___long_bench")

for task in longbench_tasks:
    # 1. Load and cache the dataset
    ds = load_dataset(
        "Xnhyacinth/LongBench",
        data_dir=task,
        split="test",
    )
    print(f"cached: {task}")

    # 2. Find ONLY the raw hash folder.
    # We filter out any folder that already has '=' in it so we don't touch previous tasks.
    subdirs = [
        d for d in os.listdir(base_cache_dir)
        if d.startswith("default-") and "=" not in d
    ]

    if subdirs:
        # subdirs[0] will be the exact hex-hash folder just created (e.g., 'default-0065b3b...')
        current_hash_dir = os.path.join(base_cache_dir, subdirs[0])
        target_dir = os.path.join(base_cache_dir, f"default-data_dir={task}")

        # 3. Rename it safely
        shutil.move(current_hash_dir, target_dir)
        print(f"Renamed cache folder to: {target_dir}")
    else:
        print(f"Warning: Could not find raw hash folder for {task}. It might already be renamed or skipped.")
