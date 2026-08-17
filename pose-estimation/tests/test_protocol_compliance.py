"""The experiment protocol as executable checks.

Each test runs the code path it governs and asserts on behaviour, not on source
text: a suite that only greps would stay green through a rename that reverses
the behaviour. Where a number lives in both the document and the code, the test
reads the document and asserts the code agrees.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from posesim.data.mpdataset import aligned_cache_payload, write_aligned_cache
from tests.test_formal_report import SIZES, _write_matrix
from tests.test_train_moveport import _segment

_ROOT = Path(__file__).resolve().parents[1]


def _small_cache(tmp_path, subjects=range(1, 5), frames=100):
    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), frames, k * 1000) for k in subjects]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    with np.load(path, allow_pickle=False) as archive:
        return path, {key: archive[key].copy() for key in archive.files}


def test_outer_test_scoring_is_off_unless_it_is_asked_for(tmp_path):
    """design.md section 6: the windowed development runner never scores an outer test."""
    path, _ = _small_cache(tmp_path)
    common = [sys.executable, str(_ROOT / "scripts" / "train_moveport.py"),
              "--cache", str(path), "--encoder", "moments", "--fold", "0", "--seed", "0",
              "--no-imu", "--epochs", "2", "--batch", "16"]
    environment = dict(os.environ, PYTHONPATH=str(_ROOT))

    default_out = tmp_path / "default"
    result = subprocess.run(common + ["--out", str(default_out)],
                            capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    record = json.loads(next(default_out.glob("*.json")).read_text())
    assert "test_mm" not in record

    asked_out = tmp_path / "asked"
    result = subprocess.run(common + ["--inspect-outer-test", "--out", str(asked_out)],
                            capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert "test_mm" in json.loads(next(asked_out.glob("*.json")).read_text())


def test_selection_never_reads_held_out_material():
    """design.md section 3: inner selection uses development participants only."""
    from scripts.train_moveport import inner_masks

    index = np.array([[str(s), "a", "n", str(i * 10), str(i * 10 + 10)]
                      for i, s in enumerate("1 2 3 4 5 6 7 8 9".split())], dtype=str)
    folds = [["1", "2"], ["3", "4"], ["5", "6"], ["7"]]
    held_out = set(folds[0])
    for inner in range(3):
        fit, val, fit_subjects, val_subjects = inner_masks(index, folds, 0, inner)
        assert not held_out & set(fit_subjects)
        assert not held_out & set(val_subjects)
        assert not set(fit_subjects) & set(val_subjects)


def test_a_streaming_report_emits_the_contract_fields(tmp_path):
    """CLAUDE.md EVAL-2 and design.md section 2: alignment convention and declared delay."""
    import torch

    from posesim.analysis.streaming import (ALIGNMENT_CONVENTION, PRESSURE_AVAILABILITY_DELAY_S,
                                            evaluate_streaming)

    _, cache = _small_cache(tmp_path, subjects=(1, 2))

    class _Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.imu_dim = 0
            self.tcn = torch.nn.Module()
            self.tcn.head = None

        def forward(self, grid, imu=None):
            b, t = grid.shape[:2]
            return torch.zeros(b, t, 10, 3), torch.zeros(b, t, 10)

    report = evaluate_streaming(_Constant(), cache, ("1", "2"),
                                (np.zeros((10, 3)), np.ones((10, 3))))
    required = {"alignment", "availability_delay_s", "strata", "frames_scored",
                "invalid_input_frames", "segments", "subjects",
                "subject_averaged_mpjpe_mm", "pooled_frame_mpjpe_mm"}
    assert required <= set(report)
    assert report["alignment"] == ALIGNMENT_CONVENTION
    assert report["availability_delay_s"] == PRESSURE_AVAILABILITY_DELAY_S
    assert set(report["strata"]) == {"activity", "marker", "phase", "locomotion"}


def test_a_tampered_contract_hash_fails_closed(tmp_path):
    """imu.md section 4: a frozen filter is a contract with a verified coefficient hash."""
    from posesim.shank_imu.signal import load_antialias_contract
    from posesim.shank_imu.state import load_state_contract

    assert len(load_antialias_contract().coefficients) == 41
    assert len(load_state_contract().fir_coefficients) == 15

    source = _ROOT / "posesim" / "shank_imu" / "antialias_contract.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["coefficients_sha256"] = "0" * 64
    tampered = tmp_path / "antialias_contract.json"
    tampered.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_antialias_contract(tampered)


def test_the_release_foot_imu_never_gates_a_formal_variant(tmp_path):
    """design.md section 2: the released foot IMU is excluded from E0--E4."""
    from posesim.data.mpdataset import AlignedSegment
    from scripts.train_moveport import load

    segment = _segment("1", "a", 100, 0)
    imu_valid = segment.foot_imu_valid.copy()
    imu_valid[40:60] = False
    imu = segment.foot_imu_si.copy()
    imu[40:60] = np.nan
    broken = AlignedSegment(
        segment.subject, segment.activity, segment.name,
        segment.pressure_pa, segment.pressure_valid, segment.force_n, segment.force_valid,
        imu, imu_valid, segment.target_m, segment.target_valid,
        segment.contact, segment.contact_valid, segment.time_s, 0.1, 0.1, 0.1)
    path = tmp_path / "cache.npz"
    write_aligned_cache(aligned_cache_payload([broken, _segment("2", "b", 100, 1000)], 2, 0),
                        path)

    gated, _, _ = load(path, uses_foot_imu=True)
    formal, _, _ = load(path, uses_foot_imu=False)
    assert len(formal["pressure"]) > len(gated["pressure"])


def test_the_report_refuses_variants_scored_on_different_participants(tmp_path):
    """design.md section 6: all held-out participants pool into one paired analysis."""
    from scripts.formal_report import build_formal_report

    matrix = tmp_path / "matrix"
    _write_matrix(matrix)
    path = matrix / "f0/shank_imu_dense/reports/test_s0.json"
    record = json.loads(path.read_text())
    record["report"]["subjects"].pop(sorted(record["report"]["subjects"])[0])
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="participant"):
        build_formal_report(matrix, n_boot=500, seed=0, insole_areas=SIZES)


def test_the_hero_video_follows_whichever_frames_are_drawn():
    """The panels may skip cache frames; the video has to skip with them."""
    from scripts.viz_hero_gif import CACHE_RATE, VIDEO_RATE, video_picks

    every = video_picks(np.arange(600, 898))
    assert every[0] == 0
    assert every[-1] == round(297 * VIDEO_RATE / CACHE_RATE)

    # the same span of real time with only every third frame drawn must cover
    # the same span of video, not a third of it
    sparse = video_picks(np.arange(600, 898, 3))
    assert sparse[-1] == every[-1]
    assert np.all(np.diff(video_picks(np.arange(600, 900, 7))) >= 0)
