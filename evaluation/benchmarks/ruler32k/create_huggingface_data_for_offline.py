# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache sparse-attention-hub's RULER32K task configs for offline jobs."""

from datasets import load_dataset

from evaluation.benchmarks.registry import DATASET_REGISTRY, RULER_32K_TASKS

huggingface_dataset_id = DATASET_REGISTRY["ruler32k"]

for subset in RULER_32K_TASKS:
    ds = load_dataset(
        huggingface_dataset_id,
        subset,
        split=subset,
    )
    print(f"cached: {subset} ({len(ds)} samples)")

print(
    "Done. Each task was cached from its same-named config and split "
    "(for example config='cwe', split='cwe')."
)
