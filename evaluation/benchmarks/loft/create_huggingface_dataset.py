# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

loft_rag_datasets = [
    "nq_32k", "nq_128k", "nq_1m",
    "hotpotqa_32k", "hotpotqa_128k", "hotpotqa_1m",
    "musique_32k", "musique_128k", "musique_1m",
    "qampari_32k", "qampari_128k", "qampari_1m",
    "quest_32k", "quest_128k", "quest_1m",
]

hf_org = "f20180301"

for subset in loft_rag_datasets:
    parts = subset.split("_")
    length = parts[-1]
    dataset = "_".join(parts[:-1])
    hf_id = f"{hf_org}/loft-rag-{dataset}-{length}"
    ds_dict = load_dataset(hf_id)
    print(f"cached: {subset} ({hf_id}) -> splits: {list(ds_dict.keys())}")
