#!/usr/bin/env python3
"""Scan saved prediction CSVs for Qwen-style thinking blocks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.textstats import think_tag_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def prediction_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.name == "predictions.csv":
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("predictions.csv"))
    return sorted(files)


def main() -> None:
    args = parse_args()
    reports = []
    for path in prediction_files(args.paths):
        rows = tagged_rows = open_tags = close_tags = unclosed_rows = 0
        with path.open(newline="", errors="replace") as source:
            for row in csv.DictReader(source):
                rows += 1
                stats = think_tag_stats(str(row.get("predicted_answer", "")))
                tagged_rows += stats["has_think_tag"]
                unclosed_rows += stats["unclosed_think"]
                open_tags += stats["think_open_tags"]
                close_tags += stats["think_close_tags"]
        reports.append(
            {
                "file": str(path),
                "rows": rows,
                "tagged_rows": tagged_rows,
                "tagged_percent": 100 * tagged_rows / rows if rows else 0.0,
                "unclosed_rows": unclosed_rows,
                "open_tags": open_tags,
                "close_tags": close_tags,
            }
        )
    total_rows = sum(report["rows"] for report in reports)
    total_tagged = sum(report["tagged_rows"] for report in reports)
    result = {
        "files": reports,
        "totals": {
            "files": len(reports),
            "rows": total_rows,
            "tagged_rows": total_tagged,
            "tagged_percent": 100 * total_tagged / total_rows if total_rows else 0.0,
            "unclosed_rows": sum(report["unclosed_rows"] for report in reports),
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
