# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where evaluation results live.

One definition, because three used to disagree: run_benchmark wrote to
``evaluation/results/rlm``, compare read ``evaluation/results``, and score read
``benchmark_artifacts/results/rlm`` -- which the README and the runbook then
documented. Scoring a fresh run out of the box printed an empty table.

Kept free of imports so every entry point can use it without a dependency.
"""

from __future__ import annotations

# The tree holding every backend's run directories. compare.py walks this.
RESULTS_ROOT = "evaluation/results"

# Where the RLM harness writes by default. A subdirectory of RESULTS_ROOT so a
# comparison against KVPress runs finds both without being pointed at each.
RLM_RESULTS_DIR = f"{RESULTS_ROOT}/rlm"
