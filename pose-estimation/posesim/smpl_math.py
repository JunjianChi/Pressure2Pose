from __future__ import annotations
import numpy as np


def heading_yaw(joints: np.ndarray, lhip_idx: int, rhip_idx: int) -> np.ndarray:
    """Per-frame yaw (rad) of the left->right hip axis in the ground plane."""
    axis = joints[:, rhip_idx, :] - joints[:, lhip_idx, :]
    return np.arctan2(axis[:, 1], axis[:, 0])


def to_root_relative_yaw_canonical(joints, pelvis_idx, lhip_idx, rhip_idx):
    joints = np.asarray(joints, dtype=float)
    rel = joints - joints[:, pelvis_idx:pelvis_idx + 1, :]
    yaw = heading_yaw(joints, lhip_idx, rhip_idx)
    c, s = np.cos(-yaw), np.sin(-yaw)
    out = rel.copy()
    x, y = rel[..., 0], rel[..., 1]
    out[..., 0] = c[:, None] * x - s[:, None] * y
    out[..., 1] = s[:, None] * x + c[:, None] * y
    return out
