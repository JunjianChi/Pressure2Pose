"""Admission tests for the segment-streaming evaluator."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from posesim.analysis.streaming import (
    evaluate_streaming,
    interpolate_targets,
    segment_predictions,
)
from posesim.data import insole as ours
from posesim.data.mpdataset import aligned_cache_payload
from posesim.model.encoder import PosePressureNet
from posesim.model.inputs import normalise, to_grid
from tests.test_train_moveport import _segment

STATS = (np.zeros((10, 3)), np.ones((10, 3)))


def _cache(segments=None):
    segments = segments or [_segment(str(k), chr(96 + k), 100, k * 1000) for k in (1, 2)]
    return aligned_cache_payload(segments, 2, 0)


class _Constant(torch.nn.Module):
    """Free-head stand-in emitting one fixed standardised prediction."""

    def __init__(self, value=0.0):
        super().__init__()
        self.imu_dim = 0
        self.tcn = torch.nn.Module()
        self.tcn.head = None
        self.value = value

    def forward(self, grid, imu=None):
        b, t = grid.shape[:2]
        mu = torch.full((b, t, 10, 3), float(self.value))
        return mu, torch.zeros(b, t, 10)


def test_streaming_matches_a_full_context_window():
    torch.manual_seed(1)
    model = PosePressureNet(encoder="moments", head="free", imu_dim=14, n_joints=10)
    model.eval()
    cache = _cache()
    predictions, input_invalid = segment_predictions(model, cache, 0, STATS)
    assert predictions.shape == (100, 10, 3)
    assert input_invalid[:12].all() and not input_invalid[12:].any()

    start = int(cache["segment_start"][0])
    pressure = np.nan_to_num(cache["pressure_pa"][start + 20:start + 84]).copy()
    force = np.nan_to_num(cache["force_n"][start + 20:start + 84]).copy()
    imu = np.nan_to_num(cache["foot_imu_si"][start + 20:start + 84]).copy()
    shape, magnitude = normalise(pressure[None], force[None])
    features = np.concatenate([imu.reshape(1, 64, -1), magnitude], -1)
    with torch.no_grad():
        window_mu, _ = model(
            to_grid(torch.as_tensor(shape, dtype=torch.float32), ours.active_mask()),
            torch.as_tensor(features, dtype=torch.float32),
        )
    assert np.allclose(predictions[83], window_mu[0, -1].numpy(), atol=1e-5)


def test_every_valid_target_frame_scores_exactly_once():
    report = evaluate_streaming(_Constant(), _cache(), ("1", "2"), STATS)
    assert report["schema"] == "streaming-eval-v1"
    assert report["segments"] == 2
    assert report["frames"] == 200
    assert report["frames_scored"] == 176
    assert report["invalid_input_frames"] == 24
    assert set(report["subjects"]) == {"1", "2"}


def test_subject_averaged_aggregation_weights_subjects_equally():
    cache = _cache()
    report = evaluate_streaming(_Constant(0.0), cache, ("1", "2"), STATS)
    expected = {}
    for subject_index, subject in enumerate(("1", "2")):
        start = int(cache["segment_start"][subject_index])
        stop = int(cache["segment_stop"][subject_index])
        target = cache["target_m"][start:stop]
        valid = np.all(cache["target_valid"][start:stop], axis=(1, 2))
        error = np.linalg.norm(target[valid], axis=-1).mean() * 1000.0
        expected[subject] = error
        assert report["subjects"][subject]["mpjpe_mm"] == pytest.approx(error, rel=1e-6)
    assert report["subject_averaged_mpjpe_mm"] == pytest.approx(
        np.mean(list(expected.values())), rel=1e-6
    )


def test_label_override_scores_against_the_supplied_targets():
    cache = _cache()
    override = np.nan_to_num(cache["target_m"]) + 0.5
    override_valid = cache["target_valid"].copy()
    report = evaluate_streaming(
        _Constant(0.0), cache, ("1",), STATS,
        targets=override, targets_valid=override_valid, label_source="native",
    )
    assert report["label_source"] == "native"
    start, stop = int(cache["segment_start"][0]), int(cache["segment_stop"][0])
    valid = np.all(override_valid[start:stop], axis=(1, 2))
    expected = np.linalg.norm(override[start:stop][valid], axis=-1).mean() * 1000.0
    assert report["subjects"]["1"]["mpjpe_mm"] == pytest.approx(expected, rel=1e-6)


def test_segment_filter_restricts_scoring_and_guards_empty_subjects():
    cache = _cache()
    report = evaluate_streaming(_Constant(), cache, ("1",), STATS, segment_indices=(0,))
    assert report["segments"] == 1
    with pytest.raises(ValueError):
        evaluate_streaming(_Constant(), cache, ("1", "2"), STATS, segment_indices=(0,))


def test_streaming_accepts_shank_imu_features_and_rejects_mismatch(tmp_path):
    from scripts.train_moveport import attach_shank_imu, shank_imu_statistics, load
    from tests.test_train_moveport import _shank_imu_cache_for, _segment as _seg
    from posesim.data.mpdataset import write_aligned_cache

    path = tmp_path / "cache.npz"
    segments = [_seg(str(k), chr(96 + k), 100, k * 1000) for k in (1, 2)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    shank_imu_dir = tmp_path / "shank_imu"
    shank_imu_dir.mkdir()
    for segment in segments:
        _shank_imu_cache_for(segment, shank_imu_dir)
    shank_imu = attach_shank_imu(path, shank_imu_dir)
    arrays, index, _ = load(path, shank_imu=shank_imu)
    mask = np.ones(int(index[-1][4]), dtype=bool)
    shank_imu_stats = shank_imu_statistics(arrays, index, mask)

    cache = _cache(segments)
    torch.manual_seed(0)
    model = PosePressureNet(encoder="moments", head="free", imu_dim=12, n_joints=10)
    model.eval()
    predictions, input_invalid = segment_predictions(
        model, cache, 0, STATS, shank_imu=shank_imu, shank_imu_stats=shank_imu_stats)
    assert predictions.shape == (100, 10, 3)
    assert input_invalid[:30].all()          # shank_imu warm-up plus the 6-frame shift
    with pytest.raises(ValueError):
        segment_predictions(model, cache, 0, STATS)   # 12-dim model, legacy 14 features


def test_block_level_pools_the_pressure_the_model_sees():
    from posesim.data.resolution import block_average

    cache = _cache()
    rng = np.random.default_rng(0)
    spatial = rng.random(cache["pressure_pa"].shape).astype(np.float32) + 1.0
    cache["pressure_pa"] = np.where(cache["pressure_valid"], spatial,
                                    cache["pressure_pa"])
    prepooled = dict(cache)
    prepooled["pressure_pa"] = block_average(
        np.nan_to_num(cache["pressure_pa"]).astype(np.float64), 4)
    torch.manual_seed(0)
    model = PosePressureNet(encoder="dense", head="free", imu_dim=14, n_joints=10)
    model.eval()
    pooled_inside, _ = segment_predictions(model, cache, 0, STATS, block=4)
    pooled_outside, _ = segment_predictions(model, prepooled, 0, STATS)
    native, _ = segment_predictions(model, cache, 0, STATS)
    assert np.array_equal(pooled_inside, pooled_outside)
    # A trained model moves by 0.11 m at block 2 and 0.61 m at block 8; an
    # untrained one moves by 5e-8 because the normalised input is 0.002-scale.
    # Pin the input instead, where the effect is unconditional.
    start = int(cache["segment_start"][0])
    raw = np.nan_to_num(cache["pressure_pa"][start:start + 64]).astype(np.float64)
    for level, floor in ((2, 0.05), (4, 0.10), (8, 0.20)):
        pooled = block_average(raw, level)
        moved = np.abs(pooled - raw).max() / raw.max()
        assert moved > floor, (level, moved)


def test_report_carries_per_segment_scores_for_the_quality_column():
    """opensim-gates G3: the QC cutoff reports beside the headline, so the
    report must expose which segment each frame came from."""
    cache = _cache()
    report = evaluate_streaming(_Constant(), cache, ("1", "2"), STATS)
    per_segment = report["per_segment"]
    assert len(per_segment) == report["segments"]
    for entry in per_segment.values():
        assert entry["frames_scored"] > 0
        assert entry["mpjpe_mm"] > 0.0
        assert entry["subject"] in {"1", "2"}
    ids = set(per_segment)
    assert ids == {f"{s}/{a}/{n}" for s, a, n in
                   zip(cache["segment_subject"], cache["segment_activity"],
                       cache["segment_name"]) if str(s) in {"1", "2"}}


def test_report_carries_every_required_stratum():
    from posesim.data.mpdataset import TARGET_MARKERS

    cache = _cache()
    report = evaluate_streaming(_Constant(), cache, ("1", "2"), STATS)
    strata = report["strata"]
    assert set(strata) == {"activity", "marker", "phase", "locomotion"}
    assert set(strata["activity"]) == {"still"}
    assert set(strata["marker"]) == set(TARGET_MARKERS)
    assert set(strata["phase"]) <= {"stance", "swing"}
    assert set(strata["locomotion"]) <= {"gait", "posture_like"}
    for name, group in strata.items():
        for label, entry in group.items():
            assert entry["frames_scored"] > 0, (name, label)
            assert entry["subject_averaged_mpjpe_mm"] > 0.0
    # every scored frame lands in exactly one activity stratum
    assert sum(e["frames_scored"] for e in strata["activity"].values()) == \
        report["frames_scored"]


def test_invalid_shank_imu_rows_are_fed_the_channel_mean(tmp_path):
    from scripts.train_moveport import attach_shank_imu, shank_imu_statistics, load
    from posesim.data.mpdataset import write_aligned_cache
    from tests.test_train_moveport import _shank_imu_cache_for, _segment as _seg

    path = tmp_path / "cache.npz"
    segments = [_seg(str(k), chr(96 + k), 100, k * 1000) for k in (1, 2)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    shank_imu_dir = tmp_path / "shank_imu"
    shank_imu_dir.mkdir()
    for segment in segments:
        _shank_imu_cache_for(segment, shank_imu_dir)
    shank_imu = attach_shank_imu(path, shank_imu_dir)
    arrays, index, _ = load(path, shank_imu=shank_imu)
    shank_imu_stats = shank_imu_statistics(arrays, index, np.ones(int(index[-1][4]), dtype=bool))

    cache = _cache(segments)
    captured = {}

    class _Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.imu_dim = 12
            self.tcn = torch.nn.Module()
            self.tcn.head = None

        def forward(self, grid, imu=None):
            captured["imu"] = imu.clone()
            b, t = grid.shape[:2]
            return torch.zeros(b, t, 10, 3), torch.zeros(b, t, 10)

    _, invalid = segment_predictions(_Capture(), cache, 0, STATS,
                                     shank_imu=shank_imu, shank_imu_stats=shank_imu_stats)
    features = captured["imu"][0].numpy()
    assert invalid[:6].all()                       # unpairable rows at the segment head
    assert np.allclose(features[invalid], 0.0)     # standardised mean, not a gravity jump


def test_fold_segments_come_from_the_subject_table_not_the_window_mask(tmp_path):
    from posesim.data.mpdataset import write_aligned_cache
    from scripts.evaluate_streaming import fold_segments
    from tests.test_train_moveport import _segment as _seg

    path = tmp_path / "cache.npz"
    segments = [_seg(str(k), chr(96 + k), 100, k * 1000) for k in range(1, 5)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key].copy() for key in archive.files}

    folds = cache["fold_subjects"]
    held_out = [s for s in map(str, folds[0]) if s]
    indices, subjects = fold_segments(cache, held_out)

    assert sorted(subjects) == sorted(held_out)
    assert len(indices) == sum(1 for s in map(str, cache["segment_subject"])
                               if s in set(held_out))
    # a modality that invalidates a whole segment must not shrink the scored set
    starved = dict(cache)
    starved["foot_imu_valid"] = np.zeros_like(cache["foot_imu_valid"])
    assert fold_segments(starved, held_out) == (indices, subjects)


def test_statistics_mask_merges_validation_for_step_budget_runs():
    from scripts.evaluate_streaming import statistics_mask

    train = np.array([True, False, False])
    val = np.array([False, True, False])
    assert np.array_equal(statistics_mask({"steps": 100}, train, val),
                          np.array([True, True, False]))
    assert np.array_equal(statistics_mask({"val_mm": 80.0}, train, val), train)


def test_interpolate_targets_respects_bracket_validity():
    source_time = np.array([0.0, 0.1, 0.2, 0.3])
    values = np.arange(4, dtype=float)[:, None, None] * np.ones((4, 2, 3))
    valid = np.ones((4, 2, 3), dtype=bool)
    valid[2] = False
    out, out_valid = interpolate_targets(values, valid, source_time,
                                         np.array([-0.05, 0.05, 0.15, 0.30]))
    assert not out_valid[0].any()
    assert out_valid[1].all() and np.allclose(out[1], 0.5)
    assert not out_valid[2].any()
    assert out_valid[3].all() and np.allclose(out[3], 3.0)
    assert np.isnan(out[~out_valid]).all()
