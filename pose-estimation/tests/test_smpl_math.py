import os, sys
import numpy as np
from posesim.smpl_math import to_root_relative_yaw_canonical


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_translation_and_yaw_invariant():
    rng = np.random.default_rng(0)
    joints = rng.normal(size=(4, 5, 3))
    # indices: pelvis=0, lhip=1, rhip=2
    base = to_root_relative_yaw_canonical(joints, 0, 1, 2)
    # translate + rotate about Z -> canonical form must be identical
    moved = (joints @ _rot_z(0.7).T) + np.array([3.0, -2.0, 1.0])
    out = to_root_relative_yaw_canonical(moved, 0, 1, 2)
    assert np.allclose(base, out, atol=1e-6)
    # pelvis maps to origin
    assert np.allclose(out[:, 0, :], 0.0, atol=1e-6)
