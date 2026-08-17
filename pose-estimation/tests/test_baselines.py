"""Admission tests for the mean-pose and nearest-pressure-retrieval baselines."""
from __future__ import annotations

import numpy as np
import pytest

from posesim.analysis.baselines import (
    evaluate_predictions,
    mean_pose_baseline,
    retrieval_predictions,
)
from posesim.data.mpdataset import aligned_cache_payload
from posesim.model.inputs import normalise
from tests.test_train_moveport import _segment


def _cache():
    return aligned_cache_payload(
        [_segment(str(k), chr(96 + k), 100, k * 1000) for k in (1, 2)], 2, 0)


def test_mean_pose_baseline_is_the_valid_frame_mean():
    cache = _cache()
    start, stop = int(cache["segment_start"][0]), int(cache["segment_stop"][0])
    train_mask = np.zeros(len(cache["time_s"]), dtype=bool)
    train_mask[start:stop] = True
    pose = mean_pose_baseline(cache, train_mask)
    valid = np.all(cache["target_valid"][start:stop], axis=(1, 2))
    expected = np.nanmean(cache["target_m"][start:stop][valid], axis=0)
    assert pose.shape == (10, 3)
    assert np.allclose(pose, expected)


def test_retrieval_returns_the_exact_match_target():
    rng = np.random.default_rng(0)
    train_pressure = rng.random((50, 2, 253)).astype(np.float32)
    train_targets = rng.normal(size=(50, 10, 3))
    train_shape, _ = normalise(train_pressure, np.zeros((50, 2), dtype=np.float32))
    queries = train_shape[[7, 31]]
    predictions = retrieval_predictions(train_shape, train_targets, queries, chunk=16)
    assert np.allclose(predictions[0], train_targets[7])
    assert np.allclose(predictions[1], train_targets[31])


def test_fold_baselines_score_both_references_on_the_held_out_set(tmp_path):
    from posesim.analysis.baselines import fold_baselines
    from posesim.data.mpdataset import write_aligned_cache

    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 100, k * 1000) for k in range(1, 5)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key].copy() for key in archive.files}

    result = fold_baselines(cache, fold=0, retrieval_stride=1)
    assert set(result) == {"mean_pose", "retrieval"}
    for name, entry in result.items():
        assert entry["schema"] == "streaming-eval-v1"
        assert entry["frames_scored"] > 0
        assert entry["subject_averaged_mpjpe_mm"] > 0.0
    held_out = [str(s) for s in np.asarray(cache["fold_subjects"])[0] if str(s)]
    assert set(result["mean_pose"]["subjects"]) == set(held_out)


def test_evaluate_predictions_scores_like_the_streaming_report():
    cache = _cache()
    predictions = np.zeros_like(np.nan_to_num(cache["target_m"]))
    report = evaluate_predictions(cache, ("1", "2"), predictions)
    assert report["schema"] == "streaming-eval-v1"
    assert report["frames_scored"] == 176
    for subject in ("1", "2"):
        start = int(cache["segment_start"][int(subject) - 1])
        stop = int(cache["segment_stop"][int(subject) - 1])
        valid = np.all(cache["target_valid"][start:stop], axis=(1, 2))
        expected = np.linalg.norm(
            cache["target_m"][start:stop][valid], axis=-1).mean() * 1000.0
        assert report["subjects"][subject]["mpjpe_mm"] == pytest.approx(expected, rel=1e-6)
