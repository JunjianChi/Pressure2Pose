from pathlib import Path

import numpy as np
import pytest
import torch

from posesim.data import insole as ours
from posesim.data.mpdataset import AlignedSegment, aligned_cache_payload, write_aligned_cache
from scripts.train_moveport import Windows, fold_masks, load, run_epoch, target_statistics

_SMPL_JOINTS = Path(__file__).parents[1] / "data" / "processed" / "smpl_joints.npz"


def _segment(subject, name, n, offset):
    pressure = np.full((n, 2, 253), 2.0, dtype=np.float32)
    valid = np.ones_like(pressure, dtype=bool)
    valid[:12] = False
    pressure[:12] = np.nan
    force_valid = valid.all(axis=2)
    force = pressure.sum(axis=2) * ours.CELL_AREA_M2
    imu = np.full((n, 2, 6), 0.5, dtype=np.float32)
    imu_valid = np.ones_like(imu, dtype=bool)
    imu[:12] = np.nan
    imu_valid[:12] = False
    target = np.broadcast_to((np.arange(n, dtype=np.float32) + offset)[:, None, None],
                             (n, 10, 3)).copy()
    target_valid = np.ones_like(target, dtype=bool)
    target[:12] = np.nan
    target_valid[:12] = False
    contact = np.ones((n, 2), dtype=np.float32)
    return AlignedSegment(subject, "still", name, pressure, valid, force, force_valid,
                          imu, imu_valid, target, target_valid, contact,
                          np.ones((n, 2), dtype=bool), np.arange(n) / 60.0, 0.1, 0.1, 0.1)


def test_v2_cache_materialises_finite_runtime_windows_without_persisting_them(tmp_path):
    path = tmp_path / "cache.npz"
    write_aligned_cache(aligned_cache_payload([_segment(str(k), chr(96 + k), 100, k * 1000)
                                                for k in range(1, 5)], 2, 0), path)

    arrays, index, folds = load(path)
    train, _, _ = fold_masks(index, folds, 0)
    mean, std = target_statistics(arrays, index, train)
    dataset = Windows(arrays, train, (mean, std))
    shape, inertial, target = dataset[0]

    assert torch.isfinite(shape).all()
    assert torch.isfinite(inertial).all()
    assert torch.isfinite(target).all()
    with np.load(path, allow_pickle=False) as archive:
        assert archive["pressure_pa"].ndim == 3
        assert not any("window" in key for key in archive.files)


def test_training_target_statistics_count_unique_frames_not_window_overlap(tmp_path):
    path = tmp_path / "cache.npz"
    write_aligned_cache(aligned_cache_payload([_segment(str(k), chr(96 + k), 100, k * 1000)
                                                for k in range(1, 5)], 2, 0), path)
    arrays, index, folds = load(path)
    train, _, _ = fold_masks(index, folds, 0)

    mean, _ = target_statistics(arrays, index, train)

    training_segments = [k for k, row in enumerate(index) if train[int(row[3]):int(row[4])].any()]
    expected_values = []
    for segment_index in training_segments:
        start = arrays["_segment_frame_start"][segment_index]
        stop = arrays["_segment_frame_stop"][segment_index]
        values = arrays["_unique_target_m"][start:stop]
        valid = arrays["_unique_target_valid"][start:stop]
        expected_values.append(values[valid].reshape(-1))
    assert mean[0, 0] == np.mean(np.concatenate(expected_values))


@pytest.mark.skipif(not _SMPL_JOINTS.is_file(), reason="untracked SMPL joint artifact absent")
def test_kinematic_head_is_supervised_in_its_own_metric_space():
    from posesim.model.encoder import PosePressureNet
    from scripts.train_moveport import to_grid

    torch.manual_seed(0)
    model = PosePressureNet(encoder="moments", head="kinematic", imu_dim=0, n_joints=10)
    model.eval()
    mask = ours.active_mask()
    shape = torch.rand(2, 8, 2, 253)
    mean = torch.full((10, 3), 0.3)
    std = torch.full((10, 3), 0.02)
    with torch.no_grad():
        markers_m, _ = model(to_grid(shape, mask))
    y = (markers_m - mean) / std

    stats = run_epoch(model, [(shape, torch.zeros(2, 8, 0), y)], mask,
                      None, None, std, "cpu", mean=mean)
    assert stats["mm"] < 1e-3


def test_load_materialises_the_requested_window_geometry(tmp_path):
    path = tmp_path / "cache.npz"
    write_aligned_cache(aligned_cache_payload([_segment(str(k), chr(96 + k), 100, k * 1000)
                                                for k in range(1, 5)], 2, 0), path)
    arrays, index, _ = load(path, window=40, stride=20)
    assert arrays["pressure"].shape[1] == 40
    # 100 frames, first 12 invalid: starts 20, 40, and 60 survive in each segment
    assert int(index[-1][4]) == 12


def test_inner_masks_are_subject_disjoint_inside_the_development_set(tmp_path):
    from scripts.train_moveport import inner_masks

    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 100, k * 1000) for k in range(1, 7)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    arrays, index, folds = load(path)
    held_out = set(folds[0])
    all_inner_val = set()
    for inner in range(3):
        fit_mask, val_mask, fit_subjects, val_subjects = inner_masks(index, folds, 0, inner)
        assert not (set(val_subjects) & set(fit_subjects))
        assert not (set(val_subjects) | set(fit_subjects)) & held_out
        assert not (fit_mask & val_mask).any()
        for row in index:
            window_range = slice(int(row[3]), int(row[4]))
            if row[0] in held_out:
                assert not fit_mask[window_range].any()
                assert not val_mask[window_range].any()
        all_inner_val |= set(val_subjects)
    development = {row[0] for row in index} - held_out
    assert all_inner_val == development


def test_subject_averaged_validation_weights_subjects_equally(tmp_path):
    import torch

    from scripts.train_moveport import subject_averaged_validation

    path = tmp_path / "cache.npz"
    segments = [_segment("1", "a", 100, 0), _segment("2", "b", 100, 1000),
                _segment("2", "c", 100, 1000)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    arrays, index, _ = load(path)
    mask = np.ones(int(index[-1][4]), dtype=bool)
    stats = (np.zeros((10, 3)), np.ones((10, 3)))

    class _Zero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tcn = torch.nn.Module()
            self.tcn.head = None

        def forward(self, grid, imu=None):
            b, t = grid.shape[:2]
            return torch.zeros(b, t, 10, 3), torch.zeros(b, t, 10)

    report = subject_averaged_validation(_Zero(), arrays, index, mask, stats, "cpu")
    per_subject = report["per_subject_mm"]
    assert set(per_subject) == {"1", "2"}
    assert report["subject_averaged_mm"] == pytest.approx(
        (per_subject["1"] + per_subject["2"]) / 2.0
    )
    assert per_subject["2"] > per_subject["1"]


def _shank_imu_cache_for(segment, tmp_path, warmup=24):
    from posesim.shank_imu.cache import shank_imu_cache_payload, write_shank_imu_cache
    from posesim.shank_imu.generator import ShankImuSegment
    from posesim.shank_imu.signal import kaiser_lowpass

    n = len(segment.time_s)
    values = np.zeros((n, 2, 6))
    values[:, :, 1] = 9.80665
    values[:, 0, 5] = np.linspace(-1.0, 1.0, n)
    values[:, 1, 5] = np.linspace(1.0, -1.0, n)
    valid = np.ones(n, dtype=bool)
    valid[:warmup] = False
    valid[-8:] = False
    physical = np.asarray(segment.time_s, dtype=np.float64).copy()
    shank_imu = ShankImuSegment(
        values=values, physical_time_s=physical,
        available_time_s=np.where(valid, physical + 0.35, np.inf),
        group_delay_s=0.35, valid=valid,
    )
    coefficients = kaiser_lowpass(taps=41, cutoff_hz=25.0, sample_rate_hz=100.0,
                                  beta=5.65326)
    payload = shank_imu_cache_payload(shank_imu, subject=segment.subject,
                                 activity=segment.activity, name=segment.name,
                                 gravity_w=np.array([0.0, 0.0, -9.80665]),
                                 antialias_coefficients=coefficients)
    out = tmp_path / f"shank_imu_{segment.subject}_{segment.activity}_{segment.name}.npz"
    write_shank_imu_cache(payload, out)
    return out


def test_shank_imu_features_pair_on_physical_time_and_gate_windows(tmp_path):
    from scripts.train_moveport import attach_shank_imu

    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 100, k * 1000) for k in (1, 2)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    shank_imu_dir = tmp_path / "shank_imu"
    shank_imu_dir.mkdir()
    for segment in segments:
        _shank_imu_cache_for(segment, shank_imu_dir)
    arrays, index, _ = load(path)
    shank_imu, shank_imu_valid = attach_shank_imu(path, shank_imu_dir)
    assert shank_imu.shape == (200, 2, 6)
    assert shank_imu_valid.shape == (200,)
    # aligned row i represents physical time_s[i] - 0.1 = shank_imu grid index i - 6
    start = 0
    row = 40
    assert shank_imu[start + row, 0, 5] == pytest.approx(
        np.linspace(-1.0, 1.0, 100)[row - 6])
    assert not shank_imu_valid[start:start + 30].any()   # shank_imu warm-up plus the 6-frame shift
    assert shank_imu_valid[start + 40]
    assert not shank_imu_valid[start + 99]               # trailing shank_imu invalid frames


def test_attach_shank_imu_fails_closed_on_a_missing_segment_cache(tmp_path):
    from scripts.train_moveport import attach_shank_imu

    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 100, k * 1000) for k in (1, 2)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    shank_imu_dir = tmp_path / "shank_imu"
    shank_imu_dir.mkdir()
    _shank_imu_cache_for(segments[0], shank_imu_dir)        # second segment deliberately absent
    with pytest.raises(ValueError, match="2/still/b"):
        attach_shank_imu(path, shank_imu_dir)


def test_window_admission_ignores_foot_imu_validity_when_it_is_not_an_input(tmp_path):
    path = tmp_path / "cache.npz"
    segment = _segment("1", "a", 100, 0)
    imu_valid = segment.foot_imu_valid.copy()
    imu_valid[40:60] = False                       # archived stream unusable mid-segment
    imu = segment.foot_imu_si.copy()
    imu[40:60] = np.nan
    broken = AlignedSegment(
        segment.subject, segment.activity, segment.name,
        segment.pressure_pa, segment.pressure_valid, segment.force_n, segment.force_valid,
        imu, imu_valid, segment.target_m, segment.target_valid,
        segment.contact, segment.contact_valid, segment.time_s, 0.1, 0.1, 0.1)
    write_aligned_cache(aligned_cache_payload(
        [broken, _segment("2", "b", 100, 1000)], 2, 0), path)

    with_imu, index_imu, _ = load(path, uses_foot_imu=True)
    without_imu, index_plain, _ = load(path, uses_foot_imu=False)

    assert len(without_imu["pressure"]) > len(with_imu["pressure"])


def test_shank_imu_windows_gate_validity_and_standardise_features(tmp_path):
    from scripts.train_moveport import attach_shank_imu, shank_imu_statistics

    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 150, k * 1000) for k in (1, 2)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    shank_imu_dir = tmp_path / "shank_imu"
    shank_imu_dir.mkdir()
    for segment in segments:
        _shank_imu_cache_for(segment, shank_imu_dir, warmup=40)
    shank_imu = attach_shank_imu(path, shank_imu_dir)
    plain, _, _ = load(path)
    arrays, index, _ = load(path, shank_imu=shank_imu)
    assert "shank_imu" in arrays
    assert arrays["shank_imu"].shape[1:] == (64, 2, 6)
    assert len(arrays["shank_imu"]) < len(plain["pressure"])   # shank_imu warm-up removes windows
    mask = np.ones(int(index[-1][4]), dtype=bool)
    mean, std = shank_imu_statistics(arrays, index, mask)
    assert mean.shape == (2, 6) and std.shape == (2, 6)
    dataset = Windows(arrays, mask, (np.zeros((10, 3)), np.ones((10, 3))),
                      shank_imu_stats=(mean, std))
    shape, inertial, _ = dataset[0]
    assert inertial.shape == (64, 12)
    standardised = (arrays["shank_imu"][0] - mean) / std
    assert np.allclose(inertial.numpy(), standardised.reshape(64, 12), atol=1e-5)


def test_shank_imu_mirror_swaps_sides_and_flips_the_correct_axes():
    from scripts.train_moveport import mirror_shank_imu

    values = np.zeros((4, 2, 6), dtype=np.float32)
    values[:, 0] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    values[:, 1] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    mirrored = mirror_shank_imu(values)
    assert np.allclose(mirrored[:, 0], [10.0, 20.0, -30.0, -40.0, -50.0, 60.0])
    assert np.allclose(mirrored[:, 1], [1.0, 2.0, -3.0, -4.0, -5.0, 6.0])


def _run_main(tmp_path, extra):
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).parents[1]
    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 100, k * 1000) for k in range(1, 7)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    out = tmp_path / "runs"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_moveport.py"),
         "--cache", str(path), "--encoder", "moments", "--fold", "0", "--seed", "0",
         "--no-imu", "--skip-test", "--epochs", "2", "--batch", "16",
         "--out", str(out)] + extra,
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(root)),
    )
    assert result.returncode == 0, result.stderr
    jsons = list(out.glob("*.json"))
    assert len(jsons) == 1
    return json.loads(jsons[0].read_text())


def test_shank_imu_mode_trains_end_to_end(tmp_path):
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).parents[1]
    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 100, k * 1000) for k in range(1, 5)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    shank_imu_dir = tmp_path / "shank_imu"
    shank_imu_dir.mkdir()
    for segment in segments:
        _shank_imu_cache_for(segment, shank_imu_dir)
    out = tmp_path / "runs"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_moveport.py"),
         "--cache", str(path), "--encoder", "moments", "--fold", "0", "--seed", "0",
         "--shank-imu-dir", str(shank_imu_dir), "--skip-test", "--epochs", "2", "--batch", "16",
         "--out", str(out)],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(root)),
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(next(out.glob("*.json")).read_text())
    assert record["tag"].endswith("_shank_imu")
    assert record["config"]["shank_imu"] is True
    assert record["val_mm"] > 0.0


def test_inner_mode_and_shank_imu_mode_compose(tmp_path):
    """Inner selection and the virtual-IMU input are separate flags that must work
    together: the subject-averaged validation path builds its own dataset."""
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).parents[1]
    path = tmp_path / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 100, k * 1000) for k in range(1, 7)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), path)
    shank_imu_dir = tmp_path / "shank_imu"
    shank_imu_dir.mkdir()
    for segment in segments:
        _shank_imu_cache_for(segment, shank_imu_dir)
    out = tmp_path / "runs"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_moveport.py"),
         "--cache", str(path), "--encoder", "moments", "--fold", "0", "--seed", "0",
         "--shank-imu-dir", str(shank_imu_dir), "--inner-fold", "1", "--epochs", "2",
         "--batch", "16", "--out", str(out)],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(root)),
    )
    assert result.returncode == 0, result.stderr[-1500:]
    record = json.loads(next(out.glob("*.json")).read_text())
    assert record["config"]["shank_imu"] is True
    assert record["inner_fold"] == 1
    assert record["val_subject_averaged_mm"] > 0.0


def test_inner_mode_selects_on_subject_averaged_validation(tmp_path):
    record = _run_main(tmp_path, ["--inner-fold", "1"])
    assert record["inner_fold"] == 1
    assert record["val_subjects"]
    assert not (set(record["val_subjects"]) & set(record["fit_subjects"]))
    assert not (set(record["val_subjects"]) | set(record["fit_subjects"])) & set(
        record["held_out"])
    assert record["val_subject_averaged_mm"] > 0.0
    assert record["best_step"] == record["steps_per_epoch"] * (
        record["history"][-1]["epoch"] + 1) or record["best_step"] > 0


def test_steps_mode_trains_for_the_exact_budget(tmp_path):
    record = _run_main(tmp_path, ["--steps", "3"])
    assert record["steps"] == 3
    assert "val_mm" not in record


def test_normalise_clips_negative_registration_noise():
    from posesim.model.inputs import normalise

    pressure = np.zeros((2, 2, 253), dtype=np.float32)
    pressure[:, 0, :10] = 4.0
    pressure[:, 1, :4] = 1e-5
    pressure[:, 1, 4:8] = -1e-5          # near-cancelling swing-foot noise
    shape, _ = normalise(pressure, np.zeros((2, 2), dtype=np.float32))
    assert shape.min() >= 0.0
    assert np.isfinite(shape).all()
    # the swing foot keeps a tiny positive share instead of a cancelled total
    assert shape[:, 1].sum() > 0.0
    assert shape[:, 1].max() < 1e-4


def test_normalise_preserves_the_interfoot_load_share():
    from scripts.train_moveport import normalise

    pressure = np.zeros((4, 2, 253), dtype=np.float32)
    pressure[:, 0, :10] = 3.0
    pressure[:, 1, :10] = 1.0
    shape, _ = normalise(pressure, np.zeros((4, 2), dtype=np.float32))

    assert np.allclose(shape.sum(axis=(1, 2)), 1.0)
    assert np.allclose(shape[:, 0].sum(axis=1), 0.75)
    assert np.allclose(shape[:, 1].sum(axis=1), 0.25)


def test_normalise_keeps_a_swing_foot_near_zero():
    from scripts.train_moveport import normalise

    pressure = np.zeros((1, 2, 253), dtype=np.float32)
    pressure[0, 0, :50] = 10.0
    pressure[0, 1, 7] = 1e-6
    shape, _ = normalise(pressure, np.zeros((1, 2), dtype=np.float32))

    assert shape[0, 1].max() < 1e-5
