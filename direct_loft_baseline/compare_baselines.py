#!/usr/bin/env python3
"""Compare direct-HF predictions with a saved KVPress no-press CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(newline="", errors="replace") as source:
        rows = list(csv.DictReader(source))
    return {(row.get("task", ""), row.get("split", ""), row.get("question", "")): row for row in rows}


def canonical_prediction(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return " ".join(value.split())
    if isinstance(parsed, list) and len(parsed) == 1:
        return " ".join(str(parsed[0]).split())
    return " ".join(value.split())


def has_think(value: str) -> bool:
    return bool(re.search(r"</?think>", value, flags=re.IGNORECASE))


def main() -> None:
    args = parse_args()
    direct = load_rows(args.direct_csv)
    reference = load_rows(args.reference_csv)
    shared = sorted(set(direct).intersection(reference))
    exact = canonical = direct_think = reference_think = 0
    for key in shared:
        direct_prediction = direct[key].get("predicted_answer", "")
        reference_prediction = reference[key].get("predicted_answer", "")
        exact += direct_prediction == reference_prediction
        canonical += canonical_prediction(direct_prediction) == canonical_prediction(reference_prediction)
        direct_think += has_think(direct_prediction)
        reference_think += has_think(reference_prediction)
    result = {
        "direct_rows": len(direct),
        "reference_rows": len(reference),
        "shared_rows": len(shared),
        "exact_prediction_matches": exact,
        "canonical_prediction_matches": canonical,
        "exact_match_percent": 100 * exact / len(shared) if shared else 0.0,
        "canonical_match_percent": 100 * canonical / len(shared) if shared else 0.0,
        "direct_think_rows": direct_think,
        "reference_think_rows": reference_think,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
