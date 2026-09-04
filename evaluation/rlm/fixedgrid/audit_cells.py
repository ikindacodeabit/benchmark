# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find fixed-chunk cells whose sub-model never actually ran.

A sub-call that does not fit in GPU memory is not raised. `_generate` returns
"[SUB-MODEL ERROR] ... did not fit" as an ordinary sub-answer, the root spends
its call budget retrying, MAX_SUB_FIT_FAILURES withdraws llm_query, and the cell
COMPLETES and writes a metrics.json. The score then describes an RLM with no
sub-model, and looks exactly like a real one.

Three independent tells, checked here because any of them alone can mislead:

* ``sub_pressed_call_fraction`` is null -- ``pop_example_stats`` returned nothing,
  so not one sub-call reached the model. The strongest signal, but absent on
  ``no_press`` cells for a legitimate reason, so it cannot be used alone.
* ``document_coverage_fraction`` is 0 -- no slice was located in the document.
  A low value is normal (the root reads one window); an exact 0 across a whole
  cell is not.
* "did not fit in GPU memory" in the transcripts, with a count. This is the
  ground truth; the metrics tells are how you notice without reading 55 files.

Exit status is 1 when any cell is flagged, so this can gate a campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

FIT_ERROR_MARKER = "did not fit in GPU memory"


def _transcript_fit_failures(run_dir: Path) -> tuple[int, int]:
    """(transcripts containing a fit failure, transcripts examined)."""
    transcripts = run_dir / "transcripts"
    if not transcripts.is_dir():
        return 0, 0
    hits = total = 0
    for path in transcripts.iterdir():
        if not path.is_file():
            continue
        total += 1
        try:
            if FIT_ERROR_MARKER in path.read_text(errors="ignore"):
                hits += 1
        except OSError:
            continue
    return hits, total


def audit(results: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(results.rglob("metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
            metrics = json.loads(metrics_path.read_text())
        except (yaml.YAMLError, json.JSONDecodeError):
            continue
        if not config.get("fixed_chunk"):
            continue
        runtime = metrics.get("runtime", {})
        budget = config.get("sub_kv_memory_budget")
        factor = config.get("compression_factor")
        pressed = runtime.get("sub_pressed_call_fraction")
        coverage = runtime.get("document_coverage_fraction")
        hits, total = _transcript_fit_failures(run_dir)

        reasons = []
        if hits:
            reasons.append(f"{hits}/{total} transcripts hit the fit error")
        # A no_press cell presses nothing by design, so a null share is only
        # suspicious where a press was actually configured.
        if pressed is None and config.get("press") != "no_press":
            reasons.append("no sub-call stats recorded (pressed share is null)")
        if coverage == 0:
            reasons.append("document coverage is exactly 0")

        rows.append(
            {
                "run": run_dir.name,
                "dataset": config.get("data_dir"),
                "budget": budget,
                "factor": factor,
                "n": (int(budget) * int(factor)) if budget and factor else None,
                "pressed": pressed,
                "coverage": coverage,
                "fit_failures": hits,
                "transcripts": total,
                "reasons": reasons,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+", help="results tree(s) to audit")
    ap.add_argument("--quiet", action="store_true", help="list only the flagged cells")
    args = ap.parse_args()

    flagged_total = 0
    for tree in args.results:
        rows = audit(tree)
        flagged = [r for r in rows if r["reasons"]]
        flagged_total += len(flagged)
        print(f"\n=== {tree} : {len(rows)} fixed-chunk cells, {len(flagged)} flagged ===")
        for row in rows:
            if args.quiet and not row["reasons"]:
                continue
            mark = "SUSPECT" if row["reasons"] else "ok     "
            print(
                f"{mark} {str(row['dataset']):<14} B={row['budget']:<7} F={row['factor']:<5} "
                f"N={str(row['n']):<8} pressed={str(row['pressed']):<6} "
                f"cov={row['coverage'] if row['coverage'] is None else round(row['coverage'], 4)!s:<8} "
                f"fitfail={row['fit_failures']}/{row['transcripts']}"
            )
            for reason in row["reasons"]:
                print(f"          - {reason}")
    return 1 if flagged_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
