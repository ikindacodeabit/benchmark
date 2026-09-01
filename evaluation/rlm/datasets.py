# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RLM dataset adapters.

Hugging Face-backed tasks use the same registry and normalized DataFrame loader
as KVPress. Synthetic RLM smoke tasks remain local because they do not exist in
the shared benchmark registry.
"""
from __future__ import annotations

import json
import os
import random
import string
from pathlib import Path

from evaluation.benchmarks.loaders import iter_benchmark_examples, load_benchmark_dataset
from evaluation.benchmarks.registry import DATASET_REGISTRY, RULER_32K_TASKS

DATA_DIR = Path(os.environ.get("RLM_DATA_DIR", os.path.expanduser("~/rlm_data")))

WORDS = (
    "ocean mountain forest river cloud stone meadow valley harbor lantern "
    "compass voyage thunder ember willow falcon marble quartz cedar prairie"
).split()


def _filler(rng: random.Random, n_chars: int) -> str:
    out, total = [], 0
    while total < n_chars:
        sent = " ".join(rng.choices(WORDS, k=rng.randint(8, 14))).capitalize() + "."
        out.append(sent)
        total += len(sent) + 1
    return " ".join(out)


def gen_niah(n_examples: int = 50, ctx_chars: int = 200_000, seed: int = 0):
    """Single needle-in-a-haystack: retrieve a planted passkey."""
    rng = random.Random(seed)
    for i in range(n_examples):
        key = "".join(rng.choices(string.digits, k=7))
        needle = f" The secret passkey is {key}. Remember it. "
        body = _filler(rng, ctx_chars)
        pos = rng.randint(0, len(body) - 1)
        context = body[:pos] + needle + body[pos:]
        yield {
            "id": f"niah-{ctx_chars}-{i}",
            "context": context,
            "question": "What is the secret passkey mentioned in the document? Reply with the number only.",
            "answers": [key],
        }


def gen_multikey(n_examples: int = 50, ctx_chars: int = 200_000, n_keys: int = 8, seed: int = 1):
    """Multi-needle aggregation: sum planted values (RULER-style, harder)."""
    rng = random.Random(seed)
    for i in range(n_examples):
        vals = [rng.randint(10, 99) for _ in range(n_keys)]
        body = _filler(rng, ctx_chars)
        for j, v in enumerate(vals):
            pos = rng.randint(0, len(body) - 1)
            body = body[:pos] + f" Asset {j} has value {v} credits. " + body[pos:]
        yield {
            "id": f"multikey-{ctx_chars}-{i}",
            "context": body,
            "question": f"There are {n_keys} assets (Asset 0..{n_keys-1}), each with a value in credits. "
            "What is the SUM of all asset values? Reply with the number only.",
            "answers": [str(sum(vals))],
        }


def load_longbench_v2(limit: int | None = None):
    """Reads the JSONL cached by slurm/download_data.sh."""
    path = DATA_DIR / "longbench_v2.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run slurm/download_data.sh on the login node first.")
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            ex = json.loads(line)
            choices = "\n".join(f"({k}) {ex[k]}" for k in ("choice_A", "choice_B", "choice_C", "choice_D") if ex.get(k))
            yield {
                "id": ex.get("_id", f"lb2-{i}"),
                "context": ex["context"],
                "question": f"{ex['question']}\n{choices}\nAnswer with the letter (A/B/C/D) only.",
                "answers": [ex["answer"]],
            }


def load_oolong(limit: int | None = None):
    path = DATA_DIR / "oolong.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run slurm/download_data.sh on the login node first.")
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            ex = json.loads(line)
            yield {
                "id": ex.get("id", f"oolong-{i}"),
                "context": ex["context"],
                "question": ex["question"],
                "answers": ex["answers"] if isinstance(ex.get("answers"), list) else [str(ex.get("answer", ""))],
            }


SYNTHETIC_TASKS = {
    "niah": lambda limit: gen_niah(n_examples=limit or 50),
    "niah-1m": lambda limit: gen_niah(n_examples=limit or 20, ctx_chars=1_000_000, seed=7),
    "multikey": lambda limit: gen_multikey(n_examples=limit or 50),
    "oolong": load_oolong,
}

DATASET_ALIASES = {"longbench_v2": "longbench-v2"}


def available_datasets() -> tuple[str, ...]:
    """Return shared benchmarks plus backward-compatible RLM task names."""
    names = set(DATASET_REGISTRY) | set(SYNTHETIC_TASKS) | set(DATASET_ALIASES)
    # This dataset needs tokenizer-dependent needle insertion in KVPress.
    names.discard("needle_in_haystack")
    return tuple(sorted(names))


def canonical_dataset_name(dataset_name: str) -> str:
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def load_examples(
    dataset_name: str,
    task: str | None = None,
    limit: int | None = None,
    split: str | None = None,
):
    """Yield backend-neutral examples from a synthetic or shared benchmark."""
    if dataset_name in SYNTHETIC_TASKS:
        if split and split != "all":
            raise ValueError(f"{dataset_name} is generated, not split into dev/test; drop --split")
        yield from SYNTHETIC_TASKS[dataset_name](limit)
        return

    canonical_name = canonical_dataset_name(dataset_name)
    if canonical_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown RLM dataset: {dataset_name!r}")
    if canonical_name == "ruler32k":
        if task is None:
            raise ValueError("ruler32k requires --data-dir with a RULER subset")
        if task not in RULER_32K_TASKS:
            raise ValueError(f"Unknown RULER-32K subset {task!r}; expected one of {RULER_32K_TASKS}")
    frame = load_benchmark_dataset(
        dataset_name=canonical_name,
        task=task,
        dataset_registry=DATASET_REGISTRY,
    )
    yield from iter_benchmark_examples(frame, canonical_name, task, limit, split)
