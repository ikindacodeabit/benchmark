# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

huggingface_dataset_id = "MaxJeblick/InfiniteBench"

tasks_to_run = [
    "passkey",
    "kv_retrieval",
    "number_string",
    "longdialogue_qa_eng",
    "longbook_qa_eng",
    "longbook_choice_eng",
    "code_run",
    "code_debug",
    "math_find",
    "math_calc",
    "longbook_sum_eng",
    "longbook_qa_chn",
]

for task in tasks_to_run:
    # config_name == task, split is always "test" for this repo
    ds = load_dataset(
        huggingface_dataset_id,
        task,
        split="test",
    )
    print(f"cached: {task} ({len(ds)} samples)")

print("Done. Cache folders are named after the config (e.g. 'passkey', 'kv_retrieval', ...).")
