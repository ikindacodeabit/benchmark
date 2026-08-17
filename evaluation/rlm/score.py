# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate RLM JSONL results into a comparison table."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def main(results_dir: str = "benchmark_artifacts/results/rlm") -> None:
    result_root = Path(results_dir)
    common_runs = []
    for metrics_path in sorted(result_root.rglob("metrics.json")):
        config_path = metrics_path.with_name("config.yaml")
        if not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        config = yaml.safe_load(config_path.read_text()) or {}
        runtime = metrics.get("runtime", {})
        common_runs.append((config, runtime))

    if common_runs:
        hdr = (
            f"{'dataset/task':<34}{'mode':<9}{'model':<36}{'n':>5}"
            f"{'match%':>8}{'tok/q':>10}{'s/q':>8}{'unfin':>7}{'err':>5}"
        )
        print(hdr)
        print("-" * len(hdr))
        for config, runtime in common_runs:
            label = f"{config.get('dataset')}/{config.get('data_dir') or 'default'}"
            print(
                f"{label:<34}{config.get('mode', ''):<9}"
                f"{config.get('root_model', ''):<36}{runtime.get('examples', 0):>5}"
                f"{100 * runtime.get('progress_match', runtime.get('accuracy', 0)):>8.1f}"
                f"{runtime.get('average_tokens', 0):>10.0f}"
                f"{runtime.get('average_latency_s', 0):>8.1f}"
                f"{runtime.get('unfinished', 0):>7}{runtime.get('errors', 0):>5}"
            )
        return

    # Backward compatibility for results produced before the common contract.
    rows = defaultdict(lambda: {"n": 0, "correct": 0, "tokens": 0, "latency": 0.0, "unfinished": 0, "errors": 0})
    for path in sorted(result_root.glob("*.jsonl")):
        variant = path.parent.name if path.parent != result_root else "root"
        parts = path.stem.split(".", 2)
        if len(parts) != 3:
            continue
        task, mode, model = parts
        key = (f"{task}__{variant}", mode, model)
        for line in open(path):
            r = json.loads(line)
            row = rows[key]
            row["n"] += 1
            row["correct"] += bool(r.get("correct"))
            row["tokens"] += r.get("tokens", 0)
            row["latency"] += r.get("latency_s", 0)
            row["unfinished"] += not r.get("finished", True)
            row["errors"] += "error" in r

    hdr = f"{'task':<16}{'mode':<9}{'model':<40}{'n':>5}{'acc%':>8}{'tok/q':>10}{'s/q':>8}{'unfin':>7}{'err':>5}"
    print(hdr)
    print("-" * len(hdr))
    for (task, mode, model), r in sorted(rows.items()):
        n = r["n"] or 1
        print(
            f"{task:<16}{mode:<9}{model:<40}{r['n']:>5}"
            f"{100*r['correct']/n:>8.1f}{r['tokens']//n:>10}{r['latency']/n:>8.1f}"
            f"{r['unfinished']:>7}{r['errors']:>5}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmark_artifacts/results/rlm")
