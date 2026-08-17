import numpy as np


def test_subject_folds_partition_every_subject_once():
    from posesim.data.mpdataset import subject_folds

    subjects = [str(k) for k in range(1, 18)]
    groups = subject_folds(subjects, 3, seed=0)
    assert len(groups) == 3
    flattened = [subject for group in groups for subject in group]
    assert sorted(flattened) == sorted(subjects)
    assert {len(group) for group in groups} <= {5, 6}
    assert subject_folds(subjects, 3, seed=0) == groups
    assert subject_folds(subjects, 3, seed=1) != groups
import pytest
import subprocess
import sys
from types import SimpleNamespace

from posesim.data import insole as ours
from posesim.data.moveport import MARKERS, MovePortNativeSegment
from posesim.data.mpdataset import (
    ALIGNED_SCHEMA_KEYS,
    AlignedSegment,
    _native_cell_area,
    align_native_segment,
    aligned_cache_payload,
    marker_target,
    project_source_mask,
    validate_aligned_cache,
    write_aligned_cache,
)
from posesim.data.timed import TimedArray
from posesim.data.windows import WindowRef, window_refs


def _timed(values, hz, unit):
    values = np.asarray(values, dtype=np.float64)
    return TimedArray(values, np.arange(len(values)) / hz, np.isfinite(values), unit,
                      "provider_frame_index/nominal_hz", hz)


def _native_segment(n=101, pressure_hz=100.0):
    pressure_n = int(round((n - 1) * pressure_hz / 100.0)) + 1
    pressure = np.full((pressure_n, 2, 31, 11), 2.0)
    markers = np.zeros((n, len(MARKERS), 3), dtype=np.float64)
    index = {name: i for i, name in enumerate(MARKERS)}
    markers[:, index["L_IAS"]] = [-0.1, 0.0, 1.0]
    markers[:, index["R_IAS"]] = [0.1, 0.0, 1.0]
    markers[:, index["M_PSIS"]] = [0.0, -0.1, 1.0]
    for k, name in enumerate((f"{side}_{part}" for part in ("FLE", "FME", "LM", "CAL", "MH1")
                              for side in ("L", "R"))):
        markers[:, index[name]] = [0.01 * k, 0.2, 0.5]
    imu = np.zeros((n, 2, 6), dtype=np.float64)
    pressure_frames = np.arange(pressure_n, dtype=np.float64)
    marker_frames = np.arange(n, dtype=np.float64)
    return MovePortNativeSegment(
        "1", "still", "1",
        _timed(pressure, pressure_hz, "psi"),
        _timed(markers, 100.0, "m"),
        _timed(imu, 100.0, "m/s^2;rad/s"),
        {"pressure": pressure_frames, "markers": marker_frames, "foot_imu": marker_frames},
    )


def _native_with_force(pressure_hz=60.0, force_hz=60.0, invalid=False, right_scale=1.0,
                       area=63.72e-6, n=101):
    native = _native_segment(n=n, pressure_hz=pressure_hz)
    count = len(native.pressure.values)
    force_values = native.pressure.values.sum(axis=(2, 3)) * 6894.757 * area
    force_values[:, 1] *= right_scale
    force_valid = np.ones_like(force_values, dtype=bool)
    if invalid:
        force_values[:60, 0] = np.nan
        force_valid[:60, 0] = False
    force = TimedArray(force_values, np.arange(count) / force_hz, force_valid, "N",
                       "provider_frame_index/nominal_hz", force_hz)
    frames = dict(native.frames)
    frames["force"] = np.arange(count, dtype=np.float64)
    return MovePortNativeSegment(native.subject, native.activity, native.segment,
                                 native.pressure, native.markers, native.foot_imu, frames, force)


def test_native_cell_area_requires_compatible_timing_and_bilateral_consistency():
    assert _native_cell_area(_native_with_force()) == pytest.approx(63.72e-6)
    assert _native_cell_area(_native_with_force(force_hz=100.0)) is None
    assert _native_cell_area(_native_with_force(invalid=True)) is None
    assert _native_cell_area(_native_with_force(right_scale=1.2)) is None


def test_project_source_mask_does_not_depend_on_requested_subject_or_activity(monkeypatch):
    import posesim.data.mpdataset as dataset

    expected = np.ones((2, 31, 11), dtype=bool)
    calls = []
    monkeypatch.setattr(dataset, "active_mask",
                        lambda root: calls.append(root) or expected)

    actual = project_source_mask("release")

    np.testing.assert_array_equal(actual, expected)
    assert calls == ["release"]


def test_qc_and_builder_use_identical_fixed_mask_alignment(monkeypatch):
    import posesim.data.mpdataset as dataset
    from scripts import qc_moveport_alignment as qc

    native = _native_with_force(n=201)
    source_mask = np.ones((2, 31, 11), dtype=bool)
    fixed_mask = lambda root: source_mask
    monkeypatch.setattr(dataset, "project_source_mask", fixed_mask)
    monkeypatch.setattr(qc, "project_source_mask", fixed_mask)
    monkeypatch.setattr(dataset, "sequence_names", lambda root, subject, activity: ["1"])
    monkeypatch.setattr(dataset, "load_native_segment", lambda *args: native)
    monkeypatch.setattr(qc, "load_native_segment", lambda *args: native)
    monkeypatch.setattr(dataset, "subject_cell_area_native", lambda root, subject: 63.72e-6)
    monkeypatch.setattr(qc, "subject_cell_area_native", lambda root, subject: 63.72e-6)
    built = dataset.build("release", activities=("still",), subject_list=["1"], verbose=False)[0]
    args = SimpleNamespace(cache=None, root="release", subject="1", activity="still", segment="1")
    checked = qc._load(args, SimpleNamespace(error=lambda message: (_ for _ in ()).throw(ValueError(message))))

    np.testing.assert_array_equal(checked["pressure_pa"], built.pressure_pa)
    np.testing.assert_array_equal(checked["pressure_valid"], built.pressure_valid)


def test_subject_native_cell_area_uses_bilateral_median_then_snaps(monkeypatch, tmp_path):
    import posesim.data.mpdataset as dataset

    streams = {"1": _native_with_force(area=62.8e-6),
               "2": _native_with_force(area=63.3e-6)}
    monkeypatch.setattr(dataset, "sequence_names",
                        lambda root, subject, activity: list(streams) if activity == "still" else [])
    monkeypatch.setattr(dataset, "load_native_segment",
                        lambda root, subject, activity, name: streams[name])
    dataset._CELL_AREA_CACHE.clear()

    area = dataset.subject_cell_area_native(tmp_path, "1")

    assert area == 63.72e-6


def _aligned(subject, name, n, value=1.0):
    pressure = np.full((n, 2, 253), value, dtype=np.float32)
    return AlignedSegment(
        subject=subject,
        activity="still",
        name=name,
        pressure_pa=pressure,
        pressure_valid=np.ones_like(pressure, dtype=bool),
        force_n=np.full((n, 2), value * 253 * ours.CELL_AREA_M2, dtype=np.float32),
        force_valid=np.ones((n, 2), dtype=bool),
        foot_imu_si=np.full((n, 2, 6), value, dtype=np.float32),
        foot_imu_valid=np.ones((n, 2, 6), dtype=bool),
        target_m=np.full((n, 10, 3), value, dtype=np.float32),
        target_valid=np.ones((n, 10, 3), dtype=bool),
        contact=np.ones((n, 2), dtype=np.float32),
        contact_valid=np.ones((n, 2), dtype=bool),
        time_s=np.arange(n) / 60.0,
        pressure_group_delay_s=0.1,
        marker_group_delay_s=0.1,
        foot_imu_group_delay_s=0.1,
    )


def test_marker_target_preserves_order_frame_and_validity():
    native = _native_segment().markers
    values = native.values.copy()
    valid = native.valid.copy()
    target_index = list(MARKERS).index("L_FLE")
    values[4, target_index, 0] = np.nan
    valid[4, target_index, 0] = False
    markers = TimedArray(values, native.time_s, valid, native.unit, native.time_basis,
                         native.nominal_hz)

    target = marker_target(markers)

    assert target.values.shape == (101, 10, 3)
    assert target.unit == "m"
    assert target.values[0, 0, 1] == pytest.approx(0.2 - (-0.1 / 3.0))
    assert not target.valid[4, 0].any()
    assert target.valid[4, 1:].all()


def test_windows_are_views_of_whole_segments():
    values = np.arange(230)
    refs = window_refs([(0, 130), (130, 230)], size=64, stride=32)

    assert refs == (WindowRef(0, 0, 64), WindowRef(0, 32, 96), WindowRef(0, 64, 128),
                    WindowRef(1, 130, 194), WindowRef(1, 162, 226))
    assert all(np.shares_memory(values[ref.start:ref.stop], values) for ref in refs)
    assert all(not (ref.start < 130 < ref.stop) for ref in refs)


def test_explicit_window_ranges_reject_gaps_between_segments():
    with pytest.raises(ValueError, match="contiguous"):
        window_refs([(0, 100), (101, 201)], size=64, stride=32)


@pytest.mark.parametrize("pressure_hz", [60.0, 100.0])
def test_aligned_native_segment_anti_aliases_and_conserves_pressure_force(pressure_hz):
    native = _native_segment(pressure_hz=pressure_hz)
    source_mask = np.ones((2, 31, 11), dtype=bool)
    area = 63.72e-6

    segment = align_native_segment(native, root="unused", source_mask=source_mask, cell_area_m2=area)

    assert segment.pressure_pa.shape == (61, 2, 253)
    valid = segment.force_valid[:, 0]
    expected = 2.0 * source_mask[0].sum() * 6894.757 * area
    np.testing.assert_allclose(segment.force_n[valid, 0], expected, rtol=2e-6)
    assert segment.pressure_group_delay_s > 0.0
    assert not segment.pressure_valid[0].any()
    assert segment.pressure_group_delay_s == segment.marker_group_delay_s == segment.foot_imu_group_delay_s
    assert np.nanmin(segment.pressure_pa) >= 0.0


def test_marker_target_invalidates_all_targets_for_missing_or_degenerate_pelvis():
    native = _native_segment().markers
    values = native.values.copy()
    valid = native.valid.copy()
    valid[3, list(MARKERS).index("M_PSIS"), 2] = False
    values[3, list(MARKERS).index("M_PSIS"), 2] = np.nan
    values[4, list(MARKERS).index("R_IAS"), :2] = values[4, list(MARKERS).index("L_IAS"), :2]

    target = marker_target(TimedArray(values, native.time_s, valid, "m", native.time_basis, 100.0))

    assert not target.valid[3].any()
    assert not target.valid[4].any()
    assert np.isnan(target.values[3:5]).all()


def test_aligned_cache_stores_each_target_frame_once():
    segments = [_aligned("1", "a", 80, 1.0), _aligned("2", "b", 70, 2.0)]

    cache = aligned_cache_payload(segments, n_folds=2, seed=0)

    assert cache["schema_version"].item() == "moveport-aligned-v2"
    assert cache["pressure_pa"].shape == (150, 2, 253)
    assert cache["segment_start"].tolist() == [0, 80]
    assert cache["segment_stop"].tolist() == [80, 150]
    pairs = [(segment_id, time) for segment_id, a, b in
             zip(cache["segment_id"], cache["segment_start"], cache["segment_stop"])
             for time in cache["time_s"][a:b]]
    assert len(pairs) == len(set(pairs)) == 150
    assert "window" not in " ".join(cache)


def test_aligned_cache_schema_is_pickle_free_and_validated(tmp_path):
    path = tmp_path / "aligned.npz"
    payload = aligned_cache_payload([_aligned("1", "a", 80), _aligned("2", "b", 70)],
                                    n_folds=2, seed=1)

    write_aligned_cache(payload, path)

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == ALIGNED_SCHEMA_KEYS
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        validate_aligned_cache(archive)
        assert archive["common_group_delay_s"].item() == 0.1
        assert archive["resample_filter"].item() == (
            "rate_calibrated_causal_gaussian_fir;half_amplitude_hz=24;history_s=0.2")


def test_cache_validator_rejects_duplicate_times_and_subject_leaking_folds():
    payload = aligned_cache_payload([_aligned("1", "a", 80), _aligned("2", "b", 70)],
                                    n_folds=2, seed=0)
    duplicate = dict(payload)
    duplicate["time_s"] = duplicate["time_s"].copy()
    duplicate["time_s"][20] = duplicate["time_s"][19]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_aligned_cache(duplicate)

    leaked = dict(payload)
    leaked["fold_subjects"] = leaked["fold_subjects"].copy()
    leaked["fold_subjects"][1, 0] = leaked["fold_subjects"][0, 0]
    with pytest.raises(ValueError, match="exactly one fold"):
        validate_aligned_cache(leaked)


def test_cache_validator_rejects_shape_mismatch_and_nonconserved_force():
    payload = aligned_cache_payload([_aligned("1", "a", 80)], n_folds=1, seed=0)
    bad_shape = dict(payload)
    bad_shape["contact_valid"] = bad_shape["contact_valid"][:-1]
    with pytest.raises(ValueError, match="contact_valid"):
        validate_aligned_cache(bad_shape)

    bad_force = dict(payload)
    bad_force["force_n"] = bad_force["force_n"].copy()
    bad_force["force_n"][10, 0] += 1.0
    with pytest.raises(ValueError, match="derived force"):
        validate_aligned_cache(bad_force)


def test_cache_validator_pins_units_clock_and_segment_identity():
    payload = aligned_cache_payload([_aligned("1", "a", 80)], n_folds=1, seed=0)
    bad_unit = dict(payload)
    bad_unit["pressure_unit"] = np.asarray("psi")
    with pytest.raises(ValueError, match="units"):
        validate_aligned_cache(bad_unit)

    bad_clock = dict(payload)
    bad_clock["target_hz"] = np.asarray(100.0)
    with pytest.raises(ValueError, match="60-Hz"):
        validate_aligned_cache(bad_clock)

    bad_identity = dict(payload)
    bad_identity["segment_id"] = np.asarray(["2/still/a"])
    with pytest.raises(ValueError, match="identity"):
        validate_aligned_cache(bad_identity)


def test_cache_validator_rejects_object_arrays_before_writing(tmp_path):
    payload = aligned_cache_payload([_aligned("1", "a", 80)], n_folds=1, seed=0)
    payload["segment_subject"] = np.asarray(["1"], dtype=object)

    with pytest.raises(ValueError, match="object"):
        validate_aligned_cache(payload)
    with pytest.raises(ValueError, match="object"):
        write_aligned_cache(payload, tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


@pytest.mark.parametrize("mutation", ["time_rank", "time_text", "bounds_float", "fold_rank", "fold_bytes"])
def test_cache_validator_rejects_unsafe_structural_dtypes(mutation):
    payload = aligned_cache_payload([_aligned("1", "a", 80)], n_folds=1, seed=0)
    if mutation == "time_rank":
        payload["time_s"] = payload["time_s"][:, None]
    elif mutation == "time_text":
        payload["time_s"] = payload["time_s"].astype(str)
    elif mutation == "bounds_float":
        payload["segment_start"] = payload["segment_start"].astype(float)
    elif mutation == "fold_rank":
        payload["fold_subjects"] = payload["fold_subjects"].ravel()
    else:
        payload["fold_subjects"] = payload["fold_subjects"].astype("S")

    with pytest.raises(ValueError, match="dtype|time_s|segment|fold_subjects"):
        validate_aligned_cache(payload)


def test_alignment_qc_cli_help_runs():
    result = subprocess.run([sys.executable, "scripts/qc_moveport_alignment.py", "--help"],
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "--cache" in result.stdout
    assert "--root" in result.stdout
