"""Kinematic signal primitives for virtual-IMU acceptance tests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ANTIALIAS_CONTRACT_PATH = Path(__file__).with_name("antialias_contract.json")


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be one finite three-vector")
    return vector


def propagate_offset_acceleration(
    body_accel_w: np.ndarray,
    angular_velocity_w: np.ndarray,
    angular_accel_w: np.ndarray,
    offset_w: np.ndarray,
) -> np.ndarray:
    """Propagate a rigid body's origin acceleration to its offset sensor point."""
    acceleration = _vector(body_accel_w, "body_accel_w")
    angular_velocity = _vector(angular_velocity_w, "angular_velocity_w")
    angular_accel = _vector(angular_accel_w, "angular_accel_w")
    offset = _vector(offset_w, "offset_w")
    return (
        acceleration
        + np.cross(angular_accel, offset)
        + np.cross(angular_velocity, np.cross(angular_velocity, offset))
    )


def specific_force(
    acceleration_w: np.ndarray, rotation_ws: np.ndarray, gravity_w: np.ndarray
) -> np.ndarray:
    """Express world linear acceleration minus gravity in the imu frame."""
    acceleration = _vector(acceleration_w, "acceleration_w")
    gravity = _vector(gravity_w, "gravity_w")
    rotation = np.asarray(rotation_ws, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_ws must be one finite 3x3 matrix")
    return rotation.T @ (acceleration - gravity)


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be one finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be an orthonormal proper rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be an orthonormal proper rotation")
    return rotation


def imu_measurement(
    *,
    rotation_wt: np.ndarray,
    body_accel_w: np.ndarray,
    angular_velocity_w: np.ndarray,
    angular_accel_w: np.ndarray,
    offset_t: np.ndarray,
    rotation_ts: np.ndarray,
    gravity_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ideal imu (specific force, angular velocity) pair in `S`."""
    body_rotation = _rotation(rotation_wt, "rotation_wt")
    installation = _rotation(rotation_ts, "rotation_ts")
    offset_w = body_rotation @ _vector(offset_t, "offset_t")
    sensing_accel_w = propagate_offset_acceleration(
        body_accel_w, angular_velocity_w, angular_accel_w, offset_w
    )
    rotation_ws = body_rotation @ installation
    force_s = specific_force(sensing_accel_w, rotation_ws, gravity_w)
    gyro_s = rotation_ws.T @ _vector(angular_velocity_w, "angular_velocity_w")
    return force_s, gyro_s


@dataclass(frozen=True)
class FixedLagSignal:
    """One past-only signal with physical and availability timestamps."""

    values: np.ndarray
    physical_time_s: np.ndarray
    available_time_s: np.ndarray
    valid: np.ndarray


def kaiser_lowpass(
    *, taps: int, cutoff_hz: float, sample_rate_hz: float, beta: float
) -> np.ndarray:
    """Construct a unit-DC-gain odd-length Kaiser-windowed low-pass FIR."""
    if not isinstance(taps, int) or taps < 3 or not taps % 2:
        raise ValueError("taps must be an odd integer of at least three")
    if not np.isfinite([cutoff_hz, sample_rate_hz, beta]).all():
        raise ValueError("filter parameters must be finite")
    if not 0.0 < cutoff_hz < sample_rate_hz / 2.0 or beta < 0.0:
        raise ValueError("filter parameters are outside their valid range")
    samples = np.arange(taps, dtype=float) - (taps - 1) / 2.0
    coefficients = (
        2.0 * cutoff_hz / sample_rate_hz
        * np.sinc(2.0 * cutoff_hz / sample_rate_hz * samples)
        * np.kaiser(taps, beta)
    )
    return coefficients / coefficients.sum()


@dataclass(frozen=True)
class AntialiasContract:
    """The frozen 100-to-60 Hz anti-alias design and its coefficient array."""

    sample_rate_hz: float
    coefficients: np.ndarray
    group_delay_s: float
    passband_hz: float
    stopband_hz: float


def load_antialias_contract(path: str | Path = _ANTIALIAS_CONTRACT_PATH) -> AntialiasContract:
    """Load the frozen anti-alias constants and verify the coefficient hash."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != "shank-imu-antialias-contract-v1":
        raise ValueError("anti-alias contract schema is not shank-imu-antialias-contract-v1")
    coefficients = kaiser_lowpass(
        taps=int(record["taps"]),
        cutoff_hz=float(record["cutoff_hz"]),
        sample_rate_hz=float(record["sample_rate_hz"]),
        beta=float(record["beta"]),
    )
    digest = hashlib.sha256(coefficients.astype(np.float64).tobytes()).hexdigest()
    if digest != record["coefficients_sha256"]:
        raise ValueError("frozen anti-alias coefficients do not match their recorded hash")
    lag_s = (int(record["taps"]) - 1) / 2.0 / float(record["sample_rate_hz"])
    if not np.isclose(float(record["group_delay_s"]), lag_s, rtol=0.0, atol=1e-12):
        raise ValueError("group_delay_s must equal the symmetric-window lookahead")
    return AntialiasContract(
        sample_rate_hz=float(record["sample_rate_hz"]),
        coefficients=coefficients,
        group_delay_s=float(record["group_delay_s"]),
        passband_hz=float(record["passband_hz"]),
        stopband_hz=float(record["stopband_hz"]),
    )


def fir_response_bounds(
    coefficients: np.ndarray,
    *,
    sample_rate_hz: float,
    passband_hz: float,
    stopband_hz: float,
) -> dict[str, float]:
    """Measure passband minimum and stopband maximum gain on a fixed dense grid."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be a non-empty finite vector")
    if not np.isfinite([sample_rate_hz, passband_hz, stopband_hz]).all():
        raise ValueError("frequency parameters must be finite")
    if not 0.0 <= passband_hz < stopband_hz <= sample_rate_hz / 2.0:
        raise ValueError("frequency bands are outside their valid range")
    frequency_hz = np.linspace(0.0, sample_rate_hz / 2.0, 65537)
    sample_index = np.arange(len(coefficients))
    response = np.abs(
        np.exp(-2j * np.pi * np.outer(frequency_hz / sample_rate_hz, sample_index))
        @ coefficients
    )
    return {
        "min_passband_gain": float(response[frequency_hz <= passband_hz].min()),
        "max_stopband_gain": float(response[frequency_hz >= stopband_hz].max()),
    }


def fixed_lag_fir(
    values: np.ndarray,
    physical_time_s: np.ndarray,
    *,
    coefficients: np.ndarray,
    lag_samples: int,
) -> FixedLagSignal:
    """Filter with past samples only and mark unobservable terminal outputs invalid."""
    values = np.asarray(values, dtype=float)
    time = np.asarray(physical_time_s, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if values.ndim < 1 or time.ndim != 1 or len(values) != len(time):
        raise ValueError("values and physical_time_s must share a sample axis")
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("physical_time_s must be strictly increasing with two samples")
    dt = np.diff(time)
    if not np.allclose(dt, dt[0], rtol=0.0, atol=1e-12):
        raise ValueError("physical_time_s must have a fixed sampling interval")
    if coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be one non-empty finite vector")
    if not isinstance(lag_samples, int) or lag_samples < 0:
        raise ValueError("lag_samples must be a non-negative integer")

    output = np.zeros_like(values, dtype=float)
    for current in range(len(values)):
        first = max(0, current - len(coefficients) + 1)
        taps = coefficients[:current - first + 1]
        samples = values[current:first - 1 if first else None:-1]
        output[current] = np.tensordot(taps, samples, axes=(0, 0))
    valid = np.ones(len(values), dtype=bool)
    valid[:len(coefficients) - 1] = False
    if lag_samples:
        valid[-lag_samples:] = False
    available = time + lag_samples * dt[0]
    return FixedLagSignal(output, time.copy(), available, valid)


def interpolate_to_grid(
    signal: FixedLagSignal, target_time_s: np.ndarray
) -> FixedLagSignal:
    """Linearly interpolate a fixed-lag signal onto a strictly increasing grid.

    A target sample becomes available with the later of its two bracketing
    source samples; targets outside the source span or touching an invalid
    source sample are invalid, zero-valued, and never available.
    """
    target = np.asarray(target_time_s, dtype=float)
    if target.ndim != 1 or not len(target) or not np.isfinite(target).all():
        raise ValueError("target_time_s must be one finite non-empty vector")
    if len(target) > 1 and np.any(np.diff(target) <= 0.0):
        raise ValueError("target_time_s must be strictly increasing")

    source_time = signal.physical_time_s
    count = len(source_time)
    later = np.searchsorted(source_time, target, side="left")
    later_index = np.clip(later, 0, count - 1)
    inside = (later < count) & (target >= source_time[0])
    exact = inside & (source_time[later_index] == target)
    earlier_index = np.where(exact, later_index, np.clip(later_index - 1, 0, count - 1))

    span = source_time[later_index] - source_time[earlier_index]
    weight = np.divide(
        target - source_time[earlier_index],
        span,
        out=np.zeros_like(target),
        where=span > 0.0,
    ).reshape((-1,) + (1,) * (signal.values.ndim - 1))
    values = (1.0 - weight) * signal.values[earlier_index] + weight * signal.values[later_index]
    available = np.maximum(
        signal.available_time_s[earlier_index], signal.available_time_s[later_index]
    )
    valid = inside & signal.valid[earlier_index] & signal.valid[later_index]
    values[~valid] = 0.0
    available[~valid] = np.inf
    return FixedLagSignal(values, target.copy(), available, valid)


def fixed_lag_window_fir(
    values: np.ndarray,
    physical_time_s: np.ndarray,
    *,
    coefficients: np.ndarray,
    history_samples: int,
    lookahead_samples: int,
) -> FixedLagSignal:
    """Filter a physical-time signal once its finite future window has arrived."""
    values = np.asarray(values, dtype=float)
    time = np.asarray(physical_time_s, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if values.ndim < 1 or time.ndim != 1 or len(values) != len(time):
        raise ValueError("values and physical_time_s must share a sample axis")
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("physical_time_s must be strictly increasing with two samples")
    dt = np.diff(time)
    if not np.allclose(dt, dt[0], rtol=0.0, atol=1e-12):
        raise ValueError("physical_time_s must have a fixed sampling interval")
    if not isinstance(history_samples, int) or history_samples < 0:
        raise ValueError("history_samples must be a non-negative integer")
    if not isinstance(lookahead_samples, int) or lookahead_samples < 0:
        raise ValueError("lookahead_samples must be a non-negative integer")
    if coefficients.ndim != 1 or len(coefficients) != history_samples + lookahead_samples + 1:
        raise ValueError("coefficients must match the declared finite window")
    if not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")

    output = np.zeros_like(values, dtype=float)
    valid = np.zeros(len(values), dtype=bool)
    for physical_index in range(history_samples, len(values) - lookahead_samples):
        window = values[
            physical_index - history_samples:physical_index + lookahead_samples + 1
        ]
        output[physical_index] = np.tensordot(coefficients, window, axes=(0, 0))
        valid[physical_index] = True
    available = time + lookahead_samples * dt[0]
    return FixedLagSignal(output, time.copy(), available, valid)
