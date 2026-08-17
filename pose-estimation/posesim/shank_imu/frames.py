"""Analytic virtual imu-to-operational-frame mappings."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ANATOMICAL_FRAME_CONTRACT_PATH = Path(__file__).with_name("anatomical_frame_contract.json")


def _proper_rotation(value: np.ndarray, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be one finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be an orthonormal proper rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be an orthonormal proper rotation")
    return rotation


@dataclass(frozen=True)
class VirtualFrame:
    """One frozen virtual imu installation and operational tibia-frame map."""

    tibia_to_anatomical: np.ndarray
    sensor_to_tibia: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "tibia_to_anatomical", _proper_rotation(self.tibia_to_anatomical, "tibia_to_anatomical"))
        object.__setattr__(self, "sensor_to_tibia", _proper_rotation(self.sensor_to_tibia, "sensor_to_tibia"))

    @property
    def sensor_to_anatomical(self) -> np.ndarray:
        """Return the composed proper rotation from imu to operational frame."""
        return self.tibia_to_anatomical @ self.sensor_to_tibia


@dataclass(frozen=True)
class AnatomicalFrameContract:
    """The versioned bilateral tibia-to-operational map and its landmark evidence."""

    tibia_to_anatomical: dict[str, np.ndarray]
    evidence: dict[str, object]
    installation: dict[str, object] | None

    def virtual_frame(self, side: str) -> VirtualFrame:
        """Return the recorded virtual frame for one side; fail closed without an installation."""
        if side not in self.tibia_to_anatomical:
            raise ValueError(f"side must be one of {sorted(self.tibia_to_anatomical)}")
        if self.installation is None:
            raise ValueError("the contract records no virtual installation")
        return VirtualFrame(
            tibia_to_anatomical=self.tibia_to_anatomical[side],
            sensor_to_tibia=np.asarray(self.installation["r_ts"][side], dtype=float),
        )

    def lever_arm(self, side: str) -> np.ndarray:
        """Return the recorded tibia-origin-to-sensing-point offset for one side."""
        if side not in self.tibia_to_anatomical:
            raise ValueError(f"side must be one of {sorted(self.tibia_to_anatomical)}")
        if self.installation is None:
            raise ValueError("the contract records no virtual installation")
        offset = np.asarray(self.installation["p_s_in_t_m"][side], dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("p_s_in_t_m must be one finite three-vector per side")
        return offset


def load_anatomical_frame_contract(path: str | Path = _ANATOMICAL_FRAME_CONTRACT_PATH) -> AnatomicalFrameContract:
    """Load and validate the machine-readable bilateral anatomical frame contract."""
    record = json.loads(Path(path).read_text())
    if record.get("schema") != "shank-imu-anatomical-frame-contract-v1":
        raise ValueError("unsupported anatomical frame contract schema")
    maps = record.get("tibia_to_anatomical", {})
    if sorted(maps) != ["left", "right"]:
        raise ValueError("tibia_to_anatomical must define exactly the left and right sides")
    tibia_to_anatomical = {
        side: _proper_rotation(np.asarray(matrix, dtype=float), f"tibia_to_anatomical[{side}]")
        for side, matrix in maps.items()
    }
    return AnatomicalFrameContract(
        tibia_to_anatomical=tibia_to_anatomical,
        evidence=record.get("evidence", {}),
        installation=record.get("installation"),
    )


def map_bilateral_imu_measurements(
    imu: np.ndarray, *, left: VirtualFrame, right: VirtualFrame
) -> np.ndarray:
    """Map `(frames, left/right, force/gyro)` imu observations to D."""
    values = np.asarray(imu, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (2, 6) or not np.isfinite(values).all():
        raise ValueError("imu must be a finite array with shape (frames, 2, 6)")
    output = np.empty_like(values)
    for side, frame in enumerate((left, right)):
        output[:, side, :3] = values[:, side, :3] @ frame.sensor_to_anatomical.T
        output[:, side, 3:] = values[:, side, 3:] @ frame.sensor_to_anatomical.T
    return output
