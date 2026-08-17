"""Admission tests for the assembled candidate C_shank generator."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from posesim.shank_imu.frames import load_anatomical_frame_contract
from posesim.shank_imu.generator import ShankImuSegment, generate_shank_imu_segment
from posesim.shank_imu.provider import AnalyticProvider
from posesim.shank_imu.signal import kaiser_lowpass

FS = 100.0
GRAVITY_Z_UP = np.array([0.0, 0.0, -9.80665])
ROTATIONAL = np.array([False, False, False, True])
BASE_POSITION = {"left": np.array([0.1, 0.9, -0.1]), "right": np.array([0.1, 0.9, 0.1])}


def _provider() -> AnalyticProvider:
    return AnalyticProvider(
        axis=np.array([0.0, 0.0, 1.0]),
        base_rotation={side: np.eye(3) for side in ("left", "right")},
        base_position=BASE_POSITION,
    )


def _coefficients() -> np.ndarray:
    return kaiser_lowpass(taps=41, cutoff_hz=25.0, sample_rate_hz=FS, beta=5.65326)


def _generate(coordinates: np.ndarray, **overrides) -> ShankImuSegment:
    arguments = dict(
        provider=_provider(),
        rotational=ROTATIONAL,
        gravity_w=GRAVITY_Z_UP,
        antialias_coefficients=_coefficients(),
    )
    arguments.update(overrides)
    time_s = np.arange(len(coordinates)) / FS
    return generate_shank_imu_segment(coordinates, time_s, **arguments)


def test_gravity_and_coefficients_have_no_defaults():
    parameters = inspect.signature(generate_shank_imu_segment).parameters
    for name in ("gravity_w", "antialias_coefficients"):
        assert parameters[name].default is inspect.Parameter.empty
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_stationary_specific_force_follows_the_supplied_gravity():
    coordinates = np.zeros((300, 4))
    segment = _generate(coordinates)
    valid = segment.valid
    assert valid.any()
    for side in range(2):
        assert np.allclose(segment.values[valid, side, :3], -GRAVITY_Z_UP, atol=1e-6)
        assert np.allclose(segment.values[valid, side, 3:], 0.0, atol=1e-6)
    y_up = np.array([0.0, -9.80665, 0.0])
    rotated = _generate(coordinates, gravity_w=y_up)
    for side in range(2):
        assert np.allclose(rotated.values[rotated.valid, side, :3], -y_up, atol=1e-6)


def test_pure_translation_gives_identical_bilateral_outputs():
    time = np.arange(600) / FS
    coordinates = np.zeros((600, 4))
    coordinates[:, 0] = 0.05 * np.sin(2.0 * np.pi * 1.0 * time)
    coordinates[:, 2] = 0.03 * np.sin(2.0 * np.pi * 0.7 * time)
    segment = _generate(coordinates)
    valid = segment.valid
    assert valid.sum() > 100
    assert np.allclose(segment.values[valid, 0], segment.values[valid, 1], atol=1e-9)
    assert np.allclose(segment.values[valid, :, 3:], 0.0, atol=1e-7)


def test_constant_rotation_recovers_gyro_and_centripetal_terms():
    rate = 2.0
    time = np.arange(600) / FS
    coordinates = np.zeros((600, 4))
    coordinates[:, 3] = rate * time
    segment = _generate(coordinates)
    contract = load_anatomical_frame_contract()
    valid = segment.valid
    assert valid.sum() > 100
    for side_index, side in enumerate(("left", "right")):
        gyro = segment.values[valid, side_index, 3:]
        assert np.allclose(gyro, [0.0, 0.0, rate], atol=1e-3)
        sensing_point = BASE_POSITION[side] + contract.lever_arm(side)
        radial = sensing_point * np.array([1.0, 1.0, 0.0])
        expected = -rate**2 * radial - GRAVITY_Z_UP
        assert np.allclose(segment.values[valid, side_index, :3], expected, atol=2e-3)


def test_declared_total_availability_delay_is_350ms():
    segment = _generate(np.zeros((300, 4)))
    assert segment.group_delay_s == pytest.approx(0.35)
    valid = segment.valid
    delay = segment.available_time_s[valid] - segment.physical_time_s[valid]
    assert np.all(delay <= 0.35 + 1e-12)
    on_source_grid = valid & np.isclose(
        segment.physical_time_s * FS, np.round(segment.physical_time_s * FS), atol=1e-9
    )
    assert on_source_grid.any()
    assert np.allclose(
        segment.available_time_s[on_source_grid]
        - segment.physical_time_s[on_source_grid],
        0.34,
        atol=1e-12,
    )


def test_prefix_invariance_of_valid_outputs():
    rng = np.random.default_rng(3)
    coordinates = np.zeros((600, 4))
    coordinates[:, :3] = np.cumsum(rng.normal(scale=0.002, size=(600, 3)), axis=0)
    full = _generate(coordinates)
    truncated = _generate(coordinates[:500])
    usable = truncated.valid
    assert usable.any()
    count = len(truncated.physical_time_s)
    assert np.array_equal(truncated.physical_time_s, full.physical_time_s[:count])
    assert np.array_equal(truncated.values[usable], full.values[:count][usable])
    assert np.array_equal(
        truncated.available_time_s[usable], full.available_time_s[:count][usable]
    )


def test_invalid_edges_follow_the_stage_arithmetic():
    segment = _generate(np.zeros((600, 4)))
    first_valid_source = (35 - 1 - 14) + 20
    last_valid_source = 600 - 1 - 14 - 20
    expected = np.zeros(len(segment.physical_time_s), dtype=bool)
    lower = first_valid_source / FS
    upper = last_valid_source / FS
    for index, target in enumerate(segment.physical_time_s):
        earlier = np.floor(target * FS + 1e-9) / FS
        later = np.ceil(target * FS - 1e-9) / FS
        expected[index] = earlier >= lower - 1e-12 and later <= upper + 1e-12
    assert np.array_equal(segment.valid, expected)
    assert np.all(segment.values[~segment.valid] == 0.0)
    assert np.all(np.isinf(segment.available_time_s[~segment.valid]))


def test_rejects_malformed_configuration():
    coordinates = np.zeros((300, 4))
    time_s = np.arange(300) / FS
    good = dict(
        provider=_provider(),
        rotational=ROTATIONAL,
        gravity_w=GRAVITY_Z_UP,
        antialias_coefficients=_coefficients(),
    )
    with pytest.raises(ValueError):
        generate_shank_imu_segment(
            coordinates, time_s, **{**good, "antialias_coefficients": np.ones(40) / 40.0}
        )
    with pytest.raises(ValueError):
        generate_shank_imu_segment(
            coordinates, time_s, **{**good, "gravity_w": np.array([0.0, np.nan, 0.0])}
        )
    with pytest.raises(ValueError):
        generate_shank_imu_segment(
            coordinates, time_s, **{**good, "rotational": np.array([True, False])}
        )
