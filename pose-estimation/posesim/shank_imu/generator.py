"""Candidate C_shank assembly from IK coordinates to a 60-Hz bilateral anatomical frame segment.

Pure orchestration over the frozen state, imu, frame, and timing primitives; the
gravity vector and anti-alias coefficients are caller-supplied, never read from a
model or hardcoded here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posesim.shank_imu.frames import AnatomicalFrameContract, load_anatomical_frame_contract, map_bilateral_imu_measurements
from posesim.shank_imu.signal import (
    FixedLagSignal,
    imu_measurement,
    fixed_lag_window_fir,
    interpolate_to_grid,
)
from posesim.shank_imu.state import StateContract, fixed_lag_state, load_state_contract

SIDES = ("left", "right")


@dataclass(frozen=True)
class ShankImuSegment:
    """One bilateral 60-Hz anatomical frame observation stream with explicit timing."""

    values: np.ndarray
    physical_time_s: np.ndarray
    available_time_s: np.ndarray
    group_delay_s: float
    valid: np.ndarray


def generate_shank_imu_segment(
    coordinates: np.ndarray,
    time_s: np.ndarray,
    *,
    provider,
    rotational: np.ndarray,
    gravity_w: np.ndarray,
    antialias_coefficients: np.ndarray,
    target_rate_hz: float = 60.0,
    state_contract: StateContract | None = None,
    anatomical_frame_contract: AnatomicalFrameContract | None = None,
) -> ShankImuSegment:
    """Generate one candidate C_shank segment from a uniform coordinate series."""
    gravity = np.asarray(gravity_w, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity_w must be one finite three-vector")
    coefficients = np.asarray(antialias_coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) < 3 or not len(coefficients) % 2:
        raise ValueError("antialias_coefficients must be one odd-length vector")
    if not np.isfinite(coefficients).all():
        raise ValueError("antialias_coefficients must be finite")
    if not np.isfinite(target_rate_hz) or target_rate_hz <= 0.0:
        raise ValueError("target_rate_hz must be one positive rate")
    if state_contract is None:
        state_contract = load_state_contract()
    if anatomical_frame_contract is None:
        anatomical_frame_contract = load_anatomical_frame_contract()

    state = fixed_lag_state(
        coordinates, time_s, rotational=rotational, contract=state_contract
    )
    time = state.q.physical_time_s
    count = len(time)
    state_valid = state.q.valid & state.u.valid & state.udot.valid

    frames = {
        side: anatomical_frame_contract.virtual_frame(side) for side in SIDES
    }
    lever_arms = {side: anatomical_frame_contract.lever_arm(side) for side in SIDES}
    imu = np.zeros((count, 2, 6), dtype=float)
    for index in np.flatnonzero(state_valid):
        kinematics = provider.frame(
            state.q.values[index], state.u.values[index], state.udot.values[index]
        )
        for side_index, side in enumerate(SIDES):
            body = kinematics[side]
            force_s, gyro_s = imu_measurement(
                rotation_wt=body.rotation_wt,
                body_accel_w=body.linear_accel_w,
                angular_velocity_w=body.angular_velocity_w,
                angular_accel_w=body.angular_accel_w,
                offset_t=lever_arms[side],
                rotation_ts=frames[side].sensor_to_tibia,
                gravity_w=gravity,
            )
            imu[index, side_index, :3] = force_s
            imu[index, side_index, 3:] = gyro_s

    operational = map_bilateral_imu_measurements(
        imu, left=frames["left"], right=frames["right"]
    )

    window = (len(coefficients) - 1) // 2
    filtered = fixed_lag_window_fir(
        operational,
        time,
        coefficients=coefficients,
        history_samples=window,
        lookahead_samples=window,
    )
    invalid_in_window = np.convolve(
        (~state_valid).astype(int), np.ones(len(coefficients), dtype=int), mode="valid"
    )
    window_valid = filtered.valid.copy()
    window_valid[window:count - window] &= invalid_in_window == 0
    composed = FixedLagSignal(
        filtered.values,
        filtered.physical_time_s,
        filtered.available_time_s + state_contract.group_delay_s,
        window_valid,
    )

    duration = time[-1] - time[0]
    grid = time[0] + np.arange(
        int(np.floor(duration * target_rate_hz + 1e-9)) + 1, dtype=float
    ) / target_rate_hz
    output = interpolate_to_grid(composed, grid)
    group_delay_s = float(
        state_contract.group_delay_s
        + (window + 1) / state_contract.sample_rate_hz
    )
    return ShankImuSegment(
        values=output.values,
        physical_time_s=output.physical_time_s,
        available_time_s=output.available_time_s,
        group_delay_s=group_delay_s,
        valid=output.valid,
    )
