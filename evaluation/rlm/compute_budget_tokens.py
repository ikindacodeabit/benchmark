# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert memory budgets (MB/GB) into KV token budgets for fixedgrid's `B`.

`evaluation/rlm/fixedgrid/run_grid.sh` takes its `BUDGETS` as raw token counts
(`--memory-budget-unit tokens`) and does `N = B * F` in bash, so `B` must
already be a token count -- passing a memory value there directly would be
silently misinterpreted as tokens. This script runs the exact same
`compute_token_budget_from_memory` conversion used everywhere else in this
project's benchmarking (`kvzip_backend.py`, the LOFT/RULER/synthetic-kv matrix
runs) to turn a list of memory budgets into the matching token budgets, so the
fixedgrid grid can be run over the SAME memory budgets already used for plain
KVzip, not a separate hand-picked token list.

Only needs the model's config and dtype (for bytes-per-token), not its
weights, so it loads on the `meta` device -- no GPU, seconds not minutes.

Usage:
    python -m evaluation.rlm.compute_budget_tokens \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --memory-budgets 100 256 400 512 750 1024 2048 --unit MB

    # feed straight into fixedgrid's run_grid.sh:
    eval "$(python -m evaluation.rlm.compute_budget_tokens --unit MB \
        --memory-budgets 100 256 400 512 750 1024 2048 | tail -1)"
    DATASETS=hotpotqa FACTORS="2 4" BUDGETS="$BUDGETS" \
        bash evaluation/rlm/fixedgrid/run_grid.sh
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument(
        "--memory-budgets",
        type=float,
        nargs="+",
        required=True,
        help="memory budgets to convert, e.g. 100 256 400 512 750 1024 2048",
    )
    ap.add_argument("--unit", default="MB", choices=["MB", "GB"], help="unit for --memory-budgets")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    from kvpress.pipeline import compute_token_budget_from_memory

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="meta", trust_remote_code=True
    )

    token_budgets = []
    print(f"{'memory':>10}  {'tokens (B)':>12}")
    for memory_budget in args.memory_budgets:
        token_budget, bytes_per_token, _ = compute_token_budget_from_memory(model, memory_budget, args.unit)
        token_budgets.append(token_budget)
        print(f"{memory_budget:>8g}{args.unit}  {token_budget:>12d}")

    print()
    print("BUDGETS=\"" + " ".join(str(t) for t in token_budgets) + "\"")


if __name__ == "__main__":
    main()
