# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the two-axis fixed-grid result renderer."""

import json

import yaml

from evaluation.rlm.fixedgrid.grid_table import collect, render_dataset


def _run(root, budget, factor, score, pressed=1.0, context=None, retained=None):
    run = root / f"b{budget}-f{factor}"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": "longbench128k",
                "data_dir": "hotpotqa",
                "fixed_chunk": True,
                "sub_kv_memory_budget": budget,
                "sub_kv_memory_budget_unit": "tokens",
                "compression_factor": factor,
            }
        )
    )
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "score": score,
                "runtime": {
                    "realized_compression_factor": factor,
                    "average_sub_context_tokens": budget * factor if context is None else context,
                    "average_sub_retained_context_tokens": budget if retained is None else retained,
                    "sub_pressed_call_fraction": pressed,
                    "document_coverage_fraction": 0.75,
                    "sub_context_tokens_on_target_fraction": 1.0,
                    "sub_slice_unlocatable_calls": 0,
                    "errors": 0,
                },
            }
        )
    )


def test_collect_and_render_fixed_grid(tmp_path):
    _run(tmp_path, 1024, 1, 40.0)
    _run(tmp_path, 1024, 2, 45.0)
    frame = collect(tmp_path)
    assert len(frame) == 2

    output = tmp_path / "tables"
    output.mkdir()
    render_dataset(frame, output, "hotpotqa")

    markdown = (output / "hotpotqa.md").read_text()
    assert "40.00 | 1.00x | 75.0% | p100%" in markdown
    assert "logical-KV-retention" in markdown
    assert (output / "hotpotqa.csv").exists()


def test_effective_factor_ignores_the_unpressed_call_artifact(tmp_path):
    """A few unpressed calls must not make an on-target cell look under-compressed.

    ``realized_compression_factor`` is ``1/(1 - mean_ratio)``, so calls that fall
    under ``press_min_tokens`` and contribute ratio 0 drag it far below the
    requested factor -- 8x reads as 5.6x at a 94% pressed share -- even though
    every pressed call compressed exactly 8x. The effective factor is a
    ratio of means, so the short calls barely move it.
    """
    _run(tmp_path, 4096, 8, 15.71, pressed=0.94, context=30720, retained=3840)
    frame = collect(tmp_path)

    row = frame.iloc[0]
    assert row["effective_compression_factor"] == 8.0

    output = tmp_path / "tables"
    output.mkdir()
    render_dataset(frame, output, "hotpotqa")
    markdown = (output / "hotpotqa.md").read_text()
    assert "8.00x" in markdown
    assert "p94%" in markdown


def _loft_run(root, name, config_overrides, metrics):
    run = root / name
    run.mkdir()
    config = {
        "dataset": "loft",
        "data_dir": "nq_1m",
        "fixed_chunk": True,
        "sub_kv_memory_budget": 13563,
        "sub_kv_memory_budget_unit": "tokens",
        "compression_factor": 8.0,
    }
    config.update(config_overrides)
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "metrics.json").write_text(json.dumps(metrics))


def test_a_loft_cell_scores_on_subspan_em(tmp_path):
    """LOFT's scorer publishes no `score` key, so reading that key directly gave
    every LOFT cell a blank. compare.DATASET_SCORE_KEY names subspan_em as LOFT's
    primary metric; the grid must report the same number compare.py would."""
    _loft_run(tmp_path, "cell", {}, {"em": 0.1, "subspan_em": 0.375, "f1": 0.2, "runtime": {}})
    frame = collect(tmp_path)
    assert len(frame) == 1
    assert frame.iloc[0]["dataset"] == "nq_1m"
    assert frame.iloc[0]["qa_f1"] == 0.375


def test_an_ordinary_loft_run_is_not_a_grid_cell(tmp_path):
    """`fixed_chunk` is what makes a run a cell. Dropping the longbench128k check
    must not sweep the plain RLM/vanilla LOFT arms into the grid."""
    _loft_run(tmp_path, "cell", {"fixed_chunk": False}, {"subspan_em": 0.4, "runtime": {}})
    assert collect(tmp_path).empty
