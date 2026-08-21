# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join KVPress and RLM runs into one comparison table.

Both backends write the same `predictions.csv` / `metrics.json` / `config.yaml`
contract and score through the same canonical scorer in `benchmarks/registry.py`,
so for a given (dataset, data_dir, model) their headline numbers are the same
measurement and can be placed side by side. This script does the placing.

Usage:
  python -m evaluation.compare                                  # all results
  python -m evaluation.compare --results-dir evaluation/results
  python -m evaluation.compare --dataset ruler32k --csv out.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd
import yaml

# Scorers disagree on what to call the headline number, and some nest it per task.
# Checked in order; the first hit wins. Both backends run the SAME scorer for a
# given dataset, so whichever key is picked is picked identically for both -- that
# symmetry, not the specific key, is what makes the columns comparable.
#
# `subspan_em` sits ahead of `f1` for LOFT specifically: it is LOFT's primary RAG
# metric AND the only one all five subsets report. Without it, nq/hotpotqa/musique
# fell through to `f1` while qampari/quest (which report `coverage`, not `f1`) had
# no key hit at all and three numeric entries, so they failed the single-numeric
# fallback below and were dropped from the comparison entirely.
SCORE_KEYS = ("average", "string_match", "exact_match", "score", "accuracy", "subspan_em", "f1")

# Reported by the RLM harness about itself; never a benchmark score.
RUNTIME_KEY = "runtime"


def _flatten(metrics: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten one level of per-task nesting, e.g. {'niah_single_1': {...}}."""
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == RUNTIME_KEY:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=""))
        else:
            flat[name] = value
    return flat


def headline_score(metrics: dict) -> tuple[Optional[str], Optional[float]]:
    """Pick the comparable number out of a scorer's metrics dict."""
    flat = _flatten(metrics)
    for key in SCORE_KEYS:
        value = flat.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    # Fall back to the only numeric entry, if the scorer reports exactly one.
    numeric = {k: v for k, v in flat.items() if isinstance(v, (int, float)) and k != "num_samples"}
    if len(numeric) == 1:
        key, value = next(iter(numeric.items()))
        return key, float(value)
    return None, None


def iter_runs(results_dir: Path) -> Iterator[tuple[Path, dict, dict]]:
    """Yield (run_dir, config, metrics) for every completed run under a tree."""
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
            config = yaml.safe_load(config_path.read_text()) or {}
        except (json.JSONDecodeError, yaml.YAMLError):
            continue
        yield run_dir, config, metrics


def describe_configuration(backend: str, config: dict) -> str:
    """A short label for the knob this run varies -- the sweep's x-axis."""
    if backend == "rlm":
        mode = config.get("mode", "rlm")
        budget = (config.get("memory_budget") or {}).get("max_context_tokens")
        return f"{mode}" if budget is None else f"{mode}@ctx{budget}"
    press = config.get("press_name", "unknown")
    if config.get("memory_budget") is not None:
        return f"{press}@{config['memory_budget']:g}{config.get('memory_budget_unit', 'GB')}"
    return f"{press}@cr{config.get('compression_ratio', 0.0):.2f}"


def build_row(run_dir: Path, config: dict, metrics: dict) -> dict[str, Any]:
    backend = config.get("backend", "kvpress")
    metric_name, score = headline_score(metrics)
    runtime = metrics.get(RUNTIME_KEY, {}) if isinstance(metrics.get(RUNTIME_KEY), dict) else {}

    data_dir = config.get("data_dir")
    if isinstance(data_dir, list):
        data_dir = "__".join(data_dir)

    row: dict[str, Any] = {
        "backend": backend,
        "dataset": config.get("dataset"),
        "task": data_dir or "default",
        "model": config.get("model"),
        "configuration": describe_configuration(backend, config),
        "metric": metric_name,
        "score": score,
    }

    # The shared cost axis. KVPress varies how much of the KV cache it keeps; RLM
    # varies how much context the root holds at once. Both answer "how many context
    # tokens were live at peak", which is what KV memory is proportional to -- so
    # this is the column to plot score against for either backend.
    if backend == "rlm":
        row["context_tokens_held"] = runtime.get("average_peak_context_tokens")
        row["total_tokens"] = runtime.get("average_tokens")
        row["examples"] = runtime.get("scored")
        row["errors"] = runtime.get("errors")
        # Only meaningful for the vanilla arm, and the reason its score may be a
        # property of the truncation limit rather than of the model.
        row["context_retained"] = runtime.get("average_context_chars_retained")
    else:
        row["context_tokens_held"] = metrics.get("average_retained_context_tokens")
        row["total_tokens"] = metrics.get("average_original_context_tokens")
        row["kv_memory_gb"] = metrics.get("average_retained_kv_memory_gb")
        row["compression_ratio"] = metrics.get("average_compression_ratio")
        row["context_retained"] = 1.0  # KVPress compresses the full context, never truncates it.

    row["run_dir"] = str(run_dir)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="evaluation/results", help="tree holding both KVPress and RLM run dirs")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--model", default=None, help="restrict to one model id")
    ap.add_argument("--csv", default=None, help="also write the joined table here")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"No results directory at {results_dir}")

    rows = [build_row(*run) for run in iter_runs(results_dir)]
    if not rows:
        raise SystemExit(f"No runs with both metrics.json and config.yaml under {results_dir}")

    frame = pd.DataFrame(rows)
    if args.dataset:
        frame = frame[frame["dataset"] == args.dataset]
    if args.model:
        frame = frame[frame["model"] == args.model]
    if frame.empty:
        raise SystemExit("No runs matched the given filters")

    frame = frame.sort_values(["dataset", "task", "model", "backend", "configuration"], na_position="last")

    display_columns = [
        c
        for c in (
            "dataset",
            "task",
            "model",
            "backend",
            "configuration",
            "metric",
            "score",
            "context_tokens_held",
            "total_tokens",
            "context_retained",
            "kv_memory_gb",
            "errors",
        )
        if c in frame.columns
    ]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame[display_columns].to_string(index=False))

    # A KVPress row and an RLM row are only comparable if they ran the same
    # scorer, which means the same dataset. Say so explicitly rather than letting
    # a reader assume every printed row belongs in one plot.
    for (dataset, task), group in frame.groupby(["dataset", "task"]):
        backends = set(group["backend"])
        if len(backends) < 2:
            print(f"\nnote: {dataset}/{task} has only {backends.pop()} runs -- nothing to compare against yet")
        elif group["metric"].nunique() > 1:
            print(
                f"\nwarning: {dataset}/{task} rows use different metrics {sorted(set(group['metric']))}; not comparable"
            )

    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
