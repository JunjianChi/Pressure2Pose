"""Admission tests for the per-segment shank-imu-cache-v1 archive."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from posesim.shank_imu.cache import (
    shank_imu_cache_payload,
    load_shank_imu_cache,
    pair_with_aligned,
    validate_shank_imu_cache,
    write_shank_imu_cache,
)
from posesim.shank_imu.generator import generate_shank_imu_segment
from posesim.shank_imu.provider import AnalyticProvider
from posesim.shank_imu.signal import kaiser_lowpass

FS = 100.0
GRAVITY = np.array([0.0, 0.0, -9.80665])
_SHANK_IMU = Path(__file__).resolve().parents[1] / "posesim" / "shank_imu"


def _segment_and_payload():
    provider = AnalyticProvider(
        axis=np.array([0.0, 0.0, 1.0]),
        base_rotation={side: np.eye(3) for side in ("left", "right")},
        base_position={
            "left": np.array([0.1, 0.9, -0.1]),
            "right": np.array([0.1, 0.9, 0.1]),
        },
    )
    time = np.arange(400) / FS
    coordinates = np.zeros((400, 4))
    coordinates[:, 0] = 0.04 * np.sin(2.0 * np.pi * 1.0 * time)
    coefficients = kaiser_lowpass(taps=41, cutoff_hz=25.0, sample_rate_hz=FS, beta=5.65326)
    segment = generate_shank_imu_segment(
        coordinates,
        time,
        provider=provider,
        rotational=np.array([False, False, False, True]),
        gravity_w=GRAVITY,
        antialias_coefficients=coefficients,
    )
    payload = shank_imu_cache_payload(
        segment,
        subject="7",
        activity="mocap_high",
        name="1",
        gravity_w=GRAVITY,
        antialias_coefficients=coefficients,
    )
    return segment, coefficients, payload


def test_payload_validates_and_round_trips(tmp_path):
    segment, _, payload = _segment_and_payload()
    assert validate_shank_imu_cache(payload)
    path = tmp_path / "shank_imu_7_mocap_high_1.npz"
    write_shank_imu_cache(payload, path)
    loaded = load_shank_imu_cache(path)
    assert str(loaded["schema_version"]) == "shank-imu-cache-v1"
    assert str(loaded["segment_id"]) == "7/mocap_high/1"
    assert tuple(map(str, loaded["side_order"])) == ("left", "right")
    valid = segment.valid
    assert np.array_equal(np.all(loaded["shank_imu_valid"], axis=(1, 2)), valid)
    assert np.allclose(loaded["shank_imu_si"][valid], segment.values[valid].astype(np.float32))
    assert np.isnan(loaded["shank_imu_si"][~valid]).all()
    assert np.array_equal(loaded["physical_time_s"], segment.physical_time_s)
    assert np.isinf(loaded["available_time_s"][~valid]).all()
    assert float(loaded["group_delay_s"]) == pytest.approx(0.35)


def test_provenance_hashes_cover_contracts_and_coefficients():
    _, coefficients, payload = _segment_and_payload()
    digest = hashlib.sha256(coefficients.astype(np.float64).tobytes()).hexdigest()
    assert str(payload["antialias_coefficients_sha256"]) == digest
    for stem in ("state_contract", "anatomical_frame_contract", "gait2392_contract"):
        recorded = str(payload[f"{stem}_sha256"])
        assert recorded == hashlib.sha256((_SHANK_IMU / f"{stem}.json").read_bytes()).hexdigest()


def test_validator_rejects_tampering():
    _, _, payload = _segment_and_payload()
    wrong_schema = dict(payload)
    wrong_schema["schema_version"] = np.asarray("shank_imu-cache-v0", dtype=str)
    with pytest.raises(ValueError):
        validate_shank_imu_cache(wrong_schema)
    wrong_hash = dict(payload)
    wrong_hash["antialias_coefficients_sha256"] = np.asarray("0" * 64, dtype=str)
    with pytest.raises(ValueError):
        validate_shank_imu_cache(wrong_hash)
    nan_valid = dict(payload)
    values = np.array(payload["shank_imu_si"])
    first_valid = int(np.flatnonzero(np.all(payload["shank_imu_valid"], axis=(1, 2)))[0])
    values[first_valid, 0, 0] = np.nan
    nan_valid["shank_imu_si"] = values
    with pytest.raises(ValueError):
        validate_shank_imu_cache(nan_valid)
    finite_invalid = dict(payload)
    values = np.array(payload["shank_imu_si"])
    first_invalid = int(np.flatnonzero(~np.all(payload["shank_imu_valid"], axis=(1, 2)))[0])
    values[first_invalid, 0, 0] = 1.0
    finite_invalid["shank_imu_si"] = values
    with pytest.raises(ValueError):
        validate_shank_imu_cache(finite_invalid)
    missing = dict(payload)
    del missing["gravity_w"]
    with pytest.raises(ValueError):
        validate_shank_imu_cache(missing)
    pickled = dict(payload)
    pickled["side_order"] = np.asarray([{"side": "left"}, {"side": "right"}], dtype=object)
    with pytest.raises(ValueError):
        validate_shank_imu_cache(pickled)


def test_loader_refuses_pickled_archives(tmp_path):
    path = tmp_path / "pickled.npz"
    np.savez(path, payload=np.asarray([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):
        load_shank_imu_cache(path)


def test_pairing_uses_physical_time_not_the_aligned_clock():
    target_hz, aligned_delay = 60.0, 0.1
    frames = 240

    def motion(physical_time):
        return np.sin(2.0 * np.pi * 1.3 * physical_time)

    aligned_time_s = np.arange(frames) / target_hz
    aligned_values = motion(aligned_time_s - aligned_delay)
    shank_imu_physical = np.arange(frames) / target_hz
    shank_imu_values = motion(shank_imu_physical)

    shank_imu_index, aligned_index = pair_with_aligned(
        shank_imu_physical, aligned_time_s, aligned_group_delay_s=aligned_delay
    )
    assert len(shank_imu_index) > 200
    assert np.allclose(
        aligned_time_s[aligned_index] - aligned_delay, shank_imu_physical[shank_imu_index],
        atol=1e-9,
    )
    paired_error = np.abs(aligned_values[aligned_index] - shank_imu_values[shank_imu_index]).max()
    assert paired_error < 1e-9
    shared = min(frames, frames)
    naive_error = np.abs(aligned_values[:shared] - shank_imu_values[:shared]).max()
    assert naive_error > 0.5
