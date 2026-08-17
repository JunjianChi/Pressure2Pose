"""Per-frame bilateral tibia body-kinematics providers behind the C_shank generator."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_GAIT2392_CONTRACT_PATH = Path(__file__).with_name("gait2392_contract.json")
SIDES = ("left", "right")


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be one finite three-vector")
    return vector


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be one finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be an orthonormal proper rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be an orthonormal proper rotation")
    return rotation


def _coordinate_vector(value: np.ndarray, name: str, count: int) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (count,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be one finite vector of {count} coordinates")
    return vector


@dataclass(frozen=True)
class BodyKinematics:
    """One rigid body's world-frame pose and kinematic rates at one physical time."""

    rotation_wt: np.ndarray
    position_w: np.ndarray
    angular_velocity_w: np.ndarray
    angular_accel_w: np.ndarray
    linear_accel_w: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_wt", _rotation(self.rotation_wt, "rotation_wt"))
        for field in ("position_w", "angular_velocity_w", "angular_accel_w", "linear_accel_w"):
            object.__setattr__(self, field, _vector(getattr(self, field), field))


class AnalyticProvider:
    """Closed-form bilateral rigid motion: q = (tx, ty, tz, theta) as one world
    translation plus one rotation about a fixed world axis through the origin."""

    def __init__(
        self,
        *,
        axis: np.ndarray,
        base_rotation: dict[str, np.ndarray],
        base_position: dict[str, np.ndarray],
    ) -> None:
        axis = _vector(axis, "axis")
        norm = np.linalg.norm(axis)
        if not np.isclose(norm, 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("axis must be one unit three-vector")
        if sorted(base_rotation) != list(SIDES) or sorted(base_position) != list(SIDES):
            raise ValueError(f"bases must define exactly the sides {SIDES}")
        self._axis = axis
        self._bases = {
            side: (
                _rotation(base_rotation[side], f"base_rotation[{side}]"),
                _vector(base_position[side], f"base_position[{side}]"),
            )
            for side in SIDES
        }

    def frame(
        self, q: np.ndarray, u: np.ndarray, udot: np.ndarray
    ) -> dict[str, BodyKinematics]:
        q = _coordinate_vector(q, "q", 4)
        u = _coordinate_vector(u, "u", 4)
        udot = _coordinate_vector(udot, "udot", 4)
        skew = np.array(
            [
                [0.0, -self._axis[2], self._axis[1]],
                [self._axis[2], 0.0, -self._axis[0]],
                [-self._axis[1], self._axis[0], 0.0],
            ]
        )
        rotation_axis = (
            np.eye(3) + np.sin(q[3]) * skew + (1.0 - np.cos(q[3])) * skew @ skew
        )
        omega = u[3] * self._axis
        alpha = udot[3] * self._axis
        frame = {}
        for side, (base_rotation, base_position) in self._bases.items():
            variant = rotation_axis @ base_position
            frame[side] = BodyKinematics(
                rotation_wt=rotation_axis @ base_rotation,
                position_w=q[:3] + variant,
                angular_velocity_w=omega,
                angular_accel_w=alpha,
                linear_accel_w=udot[:3]
                + np.cross(alpha, variant)
                + np.cross(omega, np.cross(omega, variant)),
            )
        return frame


def _contract_tibia_bodies(path: str | Path) -> dict[str, str]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    bodies = {}
    for name in record["tibia_bodies"]:
        if name.endswith("_l"):
            bodies["left"] = name
        elif name.endswith("_r"):
            bodies["right"] = name
    if sorted(bodies) != list(SIDES):
        raise ValueError("the model contract must name one tibia body per side")
    return bodies


class OpenSimTibiaProvider:
    """Pinned-runtime adapter reading kinematic bilateral tibia motion from a
    supplied (q, u, udot) state via Simbody's calcBodyAccelerationFromUDot."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        coordinate_names: tuple[str, ...],
        contract_path: str | Path = _GAIT2392_CONTRACT_PATH,
    ) -> None:
        try:
            import opensim
        except ImportError as exc:
            raise ImportError(
                "OpenSimTibiaProvider requires the pinned OpenSim preprocessing runtime"
            ) from exc
        self._opensim = opensim
        self._bodies = _contract_tibia_bodies(contract_path)
        self._model = opensim.Model(str(Path(model_path)))
        self._state = self._model.initSystem()
        self._matter = self._model.getMatterSubsystem()
        if self._state.getNQ() != self._state.getNU():
            raise ValueError("the loaded model must satisfy the accepted nq == nu identity")
        if len(coordinate_names) != self._state.getNQ():
            raise ValueError("coordinate_names must cover every generalized coordinate once")
        coordinate_set = self._model.getCoordinateSet()
        coordinates = {
            coordinate_set.get(index).getName(): coordinate_set.get(index)
            for index in range(coordinate_set.getSize())
        }
        missing = [name for name in coordinate_names if name not in coordinates]
        if missing:
            raise ValueError(f"unknown model coordinates: {missing}")
        self._names = tuple(coordinate_names)
        self._coordinates = coordinates
        self._q_index = self._probe_q_indices()

    def _probe_q_indices(self) -> dict[str, int]:
        state = self._state
        reference = np.array([state.getQ().get(i) for i in range(state.getNQ())])
        indices = {}
        for name in self._names:
            coordinate = self._coordinates[name]
            original = coordinate.getValue(state)
            coordinate.setValue(state, original + 0.5, False)
            probed = np.array([state.getQ().get(i) for i in range(state.getNQ())])
            changed = np.flatnonzero(probed != reference)
            if len(changed) != 1:
                raise ValueError(f"{name} does not map to one q slot")
            indices[name] = int(changed[0])
            coordinate.setValue(state, original, False)
        return indices

    def _vec3(self, value) -> np.ndarray:
        return np.array([value.get(index) for index in range(3)])

    def _mat3(self, value) -> np.ndarray:
        return np.array([[value.get(row, col) for col in range(3)] for row in range(3)])

    def frame(
        self, q: np.ndarray, u: np.ndarray, udot: np.ndarray
    ) -> dict[str, BodyKinematics]:
        count = len(self._names)
        q = _coordinate_vector(q, "q", count)
        u = _coordinate_vector(u, "u", count)
        udot = _coordinate_vector(udot, "udot", count)
        for column, name in enumerate(self._names):
            coordinate = self._coordinates[name]
            coordinate.setValue(self._state, float(q[column]), False)
            coordinate.setSpeedValue(self._state, float(u[column]))
        self._model.realizeVelocity(self._state)
        udot_vector = self._opensim.Vector(self._state.getNU(), 0.0)
        for column, name in enumerate(self._names):
            udot_vector.set(self._q_index[name], float(udot[column]))
        acceleration = self._opensim.VectorOfSpatialVec()
        self._matter.calcBodyAccelerationFromUDot(self._state, udot_vector, acceleration)
        frame = {}
        for side, body_name in self._bodies.items():
            body = self._model.getBodySet().get(body_name)
            spatial = acceleration.get(body.getMobilizedBodyIndex())
            velocity = body.getVelocityInGround(self._state)
            transform = body.getTransformInGround(self._state)
            frame[side] = BodyKinematics(
                rotation_wt=self._mat3(transform.R()),
                position_w=self._vec3(transform.p()),
                angular_velocity_w=self._vec3(velocity.get(0)),
                angular_accel_w=self._vec3(spatial.get(0)),
                linear_accel_w=self._vec3(spatial.get(1)),
            )
        return frame
