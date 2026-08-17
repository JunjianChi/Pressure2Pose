"""Admission tests for the per-segment IK residual reader."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.segment_fit_quality import cache_segment_keys, collect, read_marker_errors

HEADER = ("Model Marker Errors from IK\nversion=1\nnRows=3\nnColumns=4\n"
          "inDegrees=no\nendheader\n"
          "time\ttotal_squared_error\tmarker_error_RMS\tmarker_error_max\n")

SEGMENTS = [("1", "still", "1"), ("2", "treadmill_normal", "high_1")]


def _write(path, values):
    rows = "".join(f"{i * 0.01:.8f}\t0.0\t{v:.8f}\t{v * 2:.8f}\n"
                   for i, v in enumerate(values))
    path.write_text(HEADER + rows, encoding="utf-8")


def _cache(segments=SEGMENTS):
    return {"segment_subject": np.array([s for s, _, _ in segments]),
            "segment_activity": np.array([a for _, a, _ in segments]),
            "segment_name": np.array([n for _, _, n in segments])}


def test_the_rms_column_is_read_not_the_max_column(tmp_path):
    path = tmp_path / "ik_errors_1_still_1.sto"
    _write(path, [0.010, 0.020, 0.030])
    assert read_marker_errors(path) == pytest.approx([0.010, 0.020, 0.030])


def test_underscores_on_both_sides_are_split_by_the_cache_not_by_guessing(tmp_path):
    """`treadmill_normal` and `high_1` both carry underscores, so a filename
    split cannot recover the boundary; the cache's segment table can."""
    keys = cache_segment_keys(_cache())
    assert keys["2_treadmill_normal_high_1"] == "2/treadmill_normal/high_1"
    assert keys["1_still_1"] == "1/still/1"


def test_collect_reduces_each_segment_to_its_within_trial_mean(tmp_path):
    _write(tmp_path / "ik_errors_1_still_1.sto", [0.010, 0.020, 0.030])
    _write(tmp_path / "ik_errors_2_treadmill_normal_high_1.sto", [0.040, 0.040, 0.040])
    rows = collect(tmp_path, cache_segment_keys(_cache()))
    assert set(rows) == {"1/still/1", "2/treadmill_normal/high_1"}
    assert rows["1/still/1"]["mean_rms_m"] == pytest.approx(0.020)
    assert rows["1/still/1"]["max_rms_m"] == pytest.approx(0.030)
    assert rows["1/still/1"]["subject"] == "1"
    assert rows["2/treadmill_normal/high_1"]["frames"] == 3


def test_a_residual_file_outside_the_cache_is_refused(tmp_path):
    _write(tmp_path / "ik_errors_1_still_1.sto", [0.010])
    _write(tmp_path / "ik_errors_9_ghost_1.sto", [0.010])
    with pytest.raises(ValueError, match="aligned cache"):
        collect(tmp_path, cache_segment_keys(_cache()))


def test_a_cache_segment_without_a_residual_file_is_refused(tmp_path):
    """A silently absent residual leaves that segment unfiltered forever."""
    _write(tmp_path / "ik_errors_1_still_1.sto", [0.010])
    with pytest.raises(ValueError, match="no marker-error file"):
        collect(tmp_path, cache_segment_keys(_cache()))


def test_collect_refuses_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no marker-error files"):
        collect(tmp_path, cache_segment_keys(_cache()))


def test_a_residual_file_without_the_rms_column_is_refused(tmp_path):
    path = tmp_path / "ik_errors_1_still_1.sto"
    path.write_text("endheader\ntime\ttotal_squared_error\n0.0\t0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marker_error_RMS"):
        read_marker_errors(path)


def test_nonfinite_residuals_are_refused(tmp_path):
    path = tmp_path / "ik_errors_1_still_1.sto"
    _write(path, [0.010, np.nan, 0.030])
    with pytest.raises(ValueError, match="finite"):
        read_marker_errors(path)
