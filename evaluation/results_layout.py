# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where evaluation results live, and what a run directory contains.

One definition, because three used to disagree: run_benchmark wrote to
``evaluation/results/rlm``, compare read ``evaluation/results``, and score read
``benchmark_artifacts/results/rlm`` -- which the README and the runbook then
documented. Scoring a fresh run out of the box printed an empty table.

Kept to the standard library so every entry point can use it without a
dependency.
"""

from __future__ import annotations

from pathlib import Path

# Anchored on this file, not on the caller's working directory. As a relative
# string this only resolved from the repository root, so `compare.py` and
# `run_benchmark.py --out` silently pointed somewhere else when run from
# `evaluation/` -- which is where run-eval.sh puts you.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# The tree holding every backend's run directories. compare.py walks this.
RESULTS_ROOT = str(_REPOSITORY_ROOT / "evaluation" / "results")

# Where the RLM harness writes by default. A subdirectory of RESULTS_ROOT so a
# comparison against KVPress runs finds both without being pointed at each.
RLM_RESULTS_DIR = str(Path(RESULTS_ROOT) / "rlm")

# The result contract every backend writes, so a reader (and compare.py) can
# rely on the same four names regardless of which harness produced the run.
PREDICTIONS_FILENAME = "predictions.csv"
METRICS_FILENAME = "metrics.json"
CONFIG_FILENAME = "config.yaml"
README_FILENAME = "README.md"


def run_artifacts(run_dir: Path | str) -> dict[str, Path]:
    """The four contract paths inside one run directory."""
    base = Path(run_dir)
    return {
        "predictions": base / PREDICTIONS_FILENAME,
        "metrics": base / METRICS_FILENAME,
        "config": base / CONFIG_FILENAME,
        "readme": base / README_FILENAME,
    }


def is_complete(run_dir: Path | str) -> bool:
    """Whether a run directory holds a finished result.

    Predictions AND metrics: metrics.json is written last (atomically), so its
    presence is what distinguishes a completed run from an interrupted one.
    """
    paths = run_artifacts(run_dir)
    return paths["predictions"].exists() and paths["metrics"].exists()
