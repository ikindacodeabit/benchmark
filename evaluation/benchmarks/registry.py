# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared benchmark and scorer registries for every evaluation backend."""

from importlib import import_module
from typing import Callable


def _lazy_scorer(module_name: str, function_name: str = "calculate_metrics") -> Callable:
    """Delay optional metric imports until that benchmark is actually scored."""

    def score(*args, **kwargs):
        module = import_module(f"{__package__}.{module_name}.calculate_metrics")
        return getattr(module, function_name)(*args, **kwargs)

    return score


DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    # Sentinel consumed by loaders._load_longbench128k; this pack is generated
    # locally from ~/rlm_data/longbench.jsonl and is never fetched from the Hub.
    "longbench128k": "local://longbench128k",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
    "loft": "f20180301/loft-rag",
    "ruler32k": "xAlg-AI/att-hub-ruler-32k",
    "ruler64k": "ollamaweights/Ruler-64k",
    "synthetic_kv": "ollamaweights/synthetic-dataset-1208-64k",
    "synthetic_kv_32k": "ollamaweights/synthetic-dataset-1208",
}


SCORER_REGISTRY = {
    "loogle": _lazy_scorer("loogle"),
    "ruler": _lazy_scorer("ruler"),
    "zero_scrolls": _lazy_scorer("zero_scrolls"),
    "infinitebench": _lazy_scorer("infinite_bench"),
    "longbench": _lazy_scorer("longbench"),
    "longbench-e": _lazy_scorer("longbench", "calculate_metrics_e"),
    "longbench128k": _lazy_scorer("longbench128k"),
    "longbench-v2": _lazy_scorer("longbenchv2"),
    "needle_in_haystack": _lazy_scorer("needle_in_haystack"),
    "aime25": _lazy_scorer("aime25"),
    "math500": _lazy_scorer("math500"),
    "loft": _lazy_scorer("loft"),
    "ruler32k": _lazy_scorer("ruler32k"),
    "ruler64k": _lazy_scorer("ruler32k"),
    "synthetic_kv": _lazy_scorer("synthetickv64k"),
    "synthetic_kv_32k": _lazy_scorer("synthetickv32k"),
}


LONGBENCH_128K_SUBSETS = ("musique", "hotpotqa", "2wikimqa", "narrativeqa", "qasper", "triviaqa")


# LOFT ships six task categories, but only the RAG subsets have been mirrored to
# the Hub (see benchmarks/loft/create_huggingface_dataset.py). Retrieval, ICL and
# SQL have no loader here yet, so they are deliberately absent rather than listed
# and broken.
LOFT_RAG_DATASETS = ("nq", "hotpotqa", "musique", "qampari", "quest")
LOFT_LENGTHS = ("32k", "128k", "1m")
LOFT_TASKS = tuple(f"{name}_{length}" for name in LOFT_RAG_DATASETS for length in LOFT_LENGTHS)


RULER_32K_TASKS = (
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
)
