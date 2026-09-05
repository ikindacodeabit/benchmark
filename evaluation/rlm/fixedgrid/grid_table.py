# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render fixed-chunk grid results as per-dataset CSV and Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.compare import headline_score


def _score(metrics: dict[str, Any], dataset: str | None = None) -> float | None:
    """The run's headline number, chosen the way the rest of the repo chooses it.

    Delegates to compare.headline_score rather than reading `score` directly:
    only some scorers publish that key. LOFT publishes `em`/`subspan_em`/`f1`
    and no `score`, so a local lookup returned None for every LOFT cell and the
    grid rendered a table of blanks. compare.DATASET_SCORE_KEY already names
    subspan_em as LOFT's primary metric, and reusing it keeps this table and
    compare.py reporting the same number for the same run.
    """
    _, value = headline_score(metrics, dataset)
    return value


def collect(results: Path) -> pd.DataFrame:
    rows = []
    seen: set[tuple[str, int, float]] = set()
    for metrics_path in sorted(results.rglob("metrics.json")):
        config_path = metrics_path.parent / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
            metrics = json.loads(metrics_path.read_text())
        except (yaml.YAMLError, json.JSONDecodeError):
            continue
        # Any benchmark the grid can drive, not just the one it was written for:
        # run_grid.sh now takes DATASET_NAME, and a LOFT-1m grid is the same
        # instrument pointed at a longer document. `fixed_chunk` is what makes a
        # run a grid cell, so that stays the real filter.
        dataset = str(config.get("dataset") or "")
        if not config.get("fixed_chunk"):
            continue
        if str(config.get("sub_kv_memory_budget_unit", "")).lower() != "tokens":
            continue
        subset = str(config.get("data_dir"))
        budget = int(config["sub_kv_memory_budget"])
        factor = float(config["compression_factor"])
        # Retrieval depth is part of a cell's identity, not a detail: two search
        # depths of one budget/factor are two different cells, and without k in the
        # key the second one raises "duplicate completed grid cell" and kills the
        # whole table build.
        search_k = int(config.get("search_k") or 0)
        key = (subset, budget, factor, search_k)
        if key in seen:
            raise ValueError(f"duplicate completed grid cell for {subset}, B={budget}, F={factor:g}, k={search_k}")
        seen.add(key)
        runtime = metrics.get("runtime", {})
        rows.append(
            {
                "dataset": subset,
                "kv_budget_tokens": budget,
                "compression_factor": factor,
                "search_k": search_k,
                # Loose answer-string presence diagnostic, not a score ceiling.
                "gold_in_retrieved_fraction": runtime.get("gold_in_retrieved_fraction"),
                "search_hits_read": runtime.get("search_hits_read"),
                "qa_f1": _score(metrics, dataset),
                "realized_compression_factor": runtime.get("realized_compression_factor"),
                # The aggregate compression the cell actually delivered: total context
                # tokens admitted over total KV retained.
                #
                # Prefer this over realized_compression_factor, which is
                # 1/(1 - mean_ratio) and is hypersensitive at high ratios: a call
                # whose slice lands under press_min_tokens skips the press and
                # contributes ratio 0, so at F=8 a mere 6% of unpressed calls drags
                # the reported factor from 8.0 to 5.6 even though every pressed call
                # compressed exactly 8x. Those short calls barely move either sum
                # here, so this ratio-of-means stays on target.
                "effective_compression_factor": _effective_factor(runtime),
                "sub_pressed_call_fraction": runtime.get("sub_pressed_call_fraction"),
                "document_coverage_fraction": runtime.get("document_coverage_fraction"),
                "sub_context_tokens_on_target_fraction": runtime.get("sub_context_tokens_on_target_fraction"),
                "sub_slice_unlocatable_calls": runtime.get("sub_slice_unlocatable_calls"),
                "errors": runtime.get("errors"),
                "run_dir": str(metrics_path.parent),
            }
        )
    return pd.DataFrame(rows)


def _effective_factor(runtime: dict) -> float | None:
    """Context tokens admitted per KV token retained, aggregated over the cell."""
    context = runtime.get("average_sub_context_tokens")
    retained = runtime.get("average_sub_retained_context_tokens")
    if context is None or retained is None or not retained:
        return None
    return float(context) / float(retained)


def _cell(row: pd.Series | None) -> str:
    if row is None or pd.isna(row.get("qa_f1")):
        return "—"
    effective = row.get("effective_compression_factor")
    coverage = row.get("document_coverage_fraction")
    pressed = row.get("sub_pressed_call_fraction")
    effective_text = "?" if pd.isna(effective) else f"{float(effective):.2f}x"
    coverage_text = "?" if pd.isna(coverage) else f"{100 * float(coverage):.1f}%"
    pressed_text = "" if pd.isna(pressed) else f" | p{100 * float(pressed):.0f}%"
    return f"{float(row['qa_f1']):.2f} | {effective_text} | {coverage_text}{pressed_text}"


def render_dataset(frame: pd.DataFrame, output: Path, subset: str) -> None:
    budgets = sorted(int(value) for value in frame["kv_budget_tokens"].unique())
    factors = sorted(float(value) for value in frame["compression_factor"].unique())
    indexed = frame.set_index(["kv_budget_tokens", "compression_factor"])
    table_rows = []
    for budget in budgets:
        row: dict[str, Any] = {"KV budget (tokens)": budget}
        for factor in factors:
            key = (budget, factor)
            match = indexed.loc[key] if key in indexed.index else None
            row[f"{factor:g}x"] = _cell(match)
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table.to_csv(output / f"{subset}.csv", index=False)

    columns = list(table.columns)
    lines = [
        f"# {subset}: fixed-chunk × logical-KV-retention grid",
        "",
        "Cells are `qa_f1 | effective factor | document coverage | pressed-call share`.",
        "",
        "Effective factor is context tokens admitted per KV token retained. It is NOT "
        "`realized_compression_factor`, which is `1/(1 - mean_ratio)` and understates a "
        "cell badly when a few sub-calls fall under `press_min_tokens` and skip the press.",
        "KVzipPress masks evicted KV but does not free it; budgets are simulated retention, "
        "not measured memory savings.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    (output / f"{subset}.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frame = collect(args.results)
    if frame.empty:
        raise SystemExit(f"No completed fixed-grid runs under {args.results}")
    output = args.output or args.results / "grid_tables"
    output.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["dataset", "search_k", "kv_budget_tokens", "compression_factor"]).to_csv(
        output / "grid_long.csv", index=False
    )
    # One budget x factor table per (subset, retrieval depth). The k=0 tables keep
    # their original filenames, so every table written before this arm existed is
    # regenerated under the same name.
    for (subset, search_k), cell_frame in frame.groupby(["dataset", "search_k"]):
        suffix = "" if not search_k else f".k{int(search_k)}"
        render_dataset(cell_frame, output, f"{subset}{suffix}")
    print(f"wrote fixed-grid tables to {output}")


if __name__ == "__main__":
    main()
