# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongBench-128K uses the canonical LongBench task dispatch unchanged."""

from evaluation.benchmarks.longbench.calculate_metrics import calculate_metrics

__all__ = ["calculate_metrics"]
