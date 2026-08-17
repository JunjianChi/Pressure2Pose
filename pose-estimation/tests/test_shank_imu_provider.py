"""Admission tests for the bilateral tibia body-kinematics providers."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from posesim.shank_imu.provider import AnalyticProvider, BodyKinematics, OpenSimTibiaProvider

AXIS = np.array([0.0, 0.0, 1.0])
BASES = {
    "left": (np.eye(3), np.array([0.1, 0.9, -0.1])),
    "right": (np.eye(3), np.array([0.1, 0.9, 0.1])),
}


def _provider() -> AnalyticProvider:
    return AnalyticProvider(
        axis=AXIS,
        base_rotation={side: rotation for side, (rotation, _) in BASES.items()},
        base_position={side: position for side, (_, position) in BASES.items()},
    )


def test_module_import_does_not_import_opensim():
    assert "posesim.shank_imu.provider" in sys.modules
    assert "opensim" not in sys.modules


def test_opensim_adapter_fails_closed_without_the_pinned_runtime():
    with pytest.raises(ImportError, match="pinned OpenSim"):
        OpenSimTibiaProvider("missing.osim", coordinate_names=("pelvis_tx",))


def test_stationary_pose_has_base_transform_and_zero_rates():
    frame = _provider().frame(np.zeros(4), np.zeros(4), np.zeros(4))
    assert sorted(frame) == ["left", "right"]
    for side, (rotation, position) in BASES.items():
        kinematics = frame[side]
        assert isinstance(kinematics, BodyKinematics)
        assert np.allclose(kinematics.rotation_wt, rotation)
        assert np.allclose(kinematics.position_w, position)
        assert np.allclose(kinematics.angular_velocity_w, 0.0)
        assert np.allclose(kinematics.angular_accel_w, 0.0)
        assert np.allclose(kinematics.linear_accel_w, 0.0)


def test_pure_translation_passes_acceleration_through():
    q = np.array([0.2, -0.1, 0.05, 0.0])
    udot = np.array([1.5, -0.7, 0.3, 0.0])
    frame = _provider().frame(q, np.array([0.4, 0.1, -0.2, 0.0]), udot)
    for side, (rotation, position) in BASES.items():
        kinematics = frame[side]
        assert np.allclose(kinematics.rotation_wt, rotation)
        assert np.allclose(kinematics.position_w, position + q[:3])
        assert np.allclose(kinematics.linear_accel_w, udot[:3])
        assert np.allclose(kinematics.angular_velocity_w, 0.0)
        assert np.allclose(kinematics.angular_accel_w, 0.0)


def test_pure_rotation_matches_rigid_body_formulas():
    theta, rate, accel = 0.6, 1.3, -0.8
    frame = _provider().frame(
        np.array([0.0, 0.0, 0.0, theta]),
        np.array([0.0, 0.0, 0.0, rate]),
        np.array([0.0, 0.0, 0.0, accel]),
    )
    cos, sin = np.cos(theta), np.sin(theta)
    rotation_axis = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    omega, alpha = rate * AXIS, accel * AXIS
    for side, (rotation, position) in BASES.items():
        kinematics = frame[side]
        variant = rotation_axis @ position
        assert np.allclose(kinematics.rotation_wt, rotation_axis @ rotation)
        assert np.allclose(kinematics.position_w, variant)
        assert np.allclose(kinematics.angular_velocity_w, omega)
        assert np.allclose(kinematics.angular_accel_w, alpha)
        assert np.allclose(
            kinematics.linear_accel_w,
            np.cross(alpha, variant) + np.cross(omega, np.cross(omega, variant)),
        )


def test_acceleration_matches_position_second_difference():
    provider = _provider()
    step = 1e-5

    def pose(time: float) -> dict[str, np.ndarray]:
        q = np.array(
            [0.05 * np.sin(3.0 * time), 0.02 * time, -0.03 * np.cos(2.0 * time),
             0.4 * np.sin(1.7 * time)]
        )
        return {
            side: kinematics.position_w
            for side, kinematics in provider.frame(q, np.zeros(4), np.zeros(4)).items()
        }

    time = 0.37
    q = np.array([0.05 * np.sin(3.0 * time), 0.02 * time, -0.03 * np.cos(2.0 * time),
                  0.4 * np.sin(1.7 * time)])
    u = np.array([0.15 * np.cos(3.0 * time), 0.02, 0.06 * np.sin(2.0 * time),
                  0.68 * np.cos(1.7 * time)])
    udot = np.array([-0.45 * np.sin(3.0 * time), 0.0, 0.12 * np.cos(2.0 * time),
                     -1.156 * np.sin(1.7 * time)])
    early, late, centre = pose(time - step), pose(time + step), pose(time)
    frame = provider.frame(q, u, udot)
    for side in ("left", "right"):
        finite_difference = (early[side] - 2.0 * centre[side] + late[side]) / step**2
        assert np.allclose(frame[side].linear_accel_w, finite_difference, atol=1e-4)


def test_rejects_malformed_inputs():
    provider = _provider()
    with pytest.raises(ValueError):
        provider.frame(np.zeros(3), np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        provider.frame(np.zeros(4), np.zeros(4), np.array([0.0, 0.0, 0.0, np.nan]))
    with pytest.raises(ValueError):
        AnalyticProvider(
            axis=np.zeros(3),
            base_rotation={side: np.eye(3) for side in ("left", "right")},
            base_position={side: np.zeros(3) for side in ("left", "right")},
        )
    with pytest.raises(ValueError):
        AnalyticProvider(
            axis=AXIS,
            base_rotation={"left": np.eye(3)},
            base_position={"left": np.zeros(3)},
        )
