# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the silent-fit-failure auditor."""

import json

import yaml

from evaluation.rlm.fixedgrid.audit_cells import audit


def _cell(root, name, *, press="kvzip", pressed=1.0, coverage=0.05, transcripts=(), fixed=True):
    run = root / name
    (run / "transcripts").mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": "loft",
                "data_dir": "nq_1m",
                "fixed_chunk": fixed,
                "press": press,
                "sub_kv_memory_budget": 27126,
                "compression_factor": 8.0,
            }
        )
    )
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "subspan_em": 0.05,
                "runtime": {"sub_pressed_call_fraction": pressed, "document_coverage_fraction": coverage},
            }
        )
    )
    for i, body in enumerate(transcripts):
        (run / "transcripts" / f"ex{i}.txt").write_text(body)


def test_a_healthy_cell_is_not_flagged(tmp_path):
    _cell(tmp_path, "good", transcripts=["all fine", "also fine"])
    (row,) = audit(tmp_path)
    assert row["reasons"] == []


def test_a_cell_whose_sub_model_never_ran_is_flagged(tmp_path):
    """The exact shape of the LOFT-1m loss: the run completed and scored, but no
    sub-call reached the model."""
    _cell(
        tmp_path,
        "bad",
        pressed=None,
        coverage=0.0,
        transcripts=["... did not fit in GPU memory ...", "clean"],
    )
    (row,) = audit(tmp_path)
    assert row["fit_failures"] == 1
    assert len(row["reasons"]) == 3


def test_a_no_press_control_is_not_flagged_for_a_null_pressed_share(tmp_path):
    """no_press presses nothing by design, so a null share is not evidence there
    -- flagging it would condemn every F=1 control in the grid."""
    _cell(tmp_path, "control", press="no_press", pressed=None, transcripts=["fine"])
    (row,) = audit(tmp_path)
    assert row["reasons"] == []


def test_transcripts_are_counted_not_just_detected(tmp_path):
    _cell(tmp_path, "partial", transcripts=["did not fit in GPU memory", "ok", "did not fit in GPU memory"])
    (row,) = audit(tmp_path)
    assert (row["fit_failures"], row["transcripts"]) == (2, 3)


def test_a_non_grid_run_is_ignored(tmp_path):
    _cell(tmp_path, "plain", fixed=False, transcripts=["did not fit in GPU memory"])
    assert audit(tmp_path) == []
