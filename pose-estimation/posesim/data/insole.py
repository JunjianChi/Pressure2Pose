from __future__ import annotations
import numpy as np

ROWS, COLS, N_SENSORS = 33, 15, 253

# Pitch of the array, and therefore the area of foot each cell speaks for. The electrode itself is
# a 3 mm-radius disc and covers only 28.3 mm2, but a cell's share of the load is its tributary area,
# not its conductive area -- so the pitch is what converts a pressure into a force.
PITCH_M = 0.0075
CELL_AREA_M2 = PITCH_M ** 2

# 1 = a sensor sits here. Same array as the hardware repo (253 ones).
IMAGE_PATTERN = np.array([
    [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],[0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],[0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],[0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],[0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
], dtype=np.int64)


_SENSOR_IDX = np.argwhere(IMAGE_PATTERN == 1)


def sensor_indices() -> np.ndarray:
    return _SENSOR_IDX


def active_mask() -> np.ndarray:
    return IMAGE_PATTERN == 1


def for_display(grid: np.ndarray, side: str) -> np.ndarray:
    """Flip a canonical-frame image so it reads as the foot it belongs to.

    Everything upstream works in one canonical foot frame -- both feet map into the same image so a
    single mask, model and forward renderer serve both. A real pair of insoles are mirror images of
    each other, so drawing both feet from the canonical frame prints two identical outlines and the
    picture silently lies about which foot is which. ``IMAGE_PATTERN`` is the left insole, so the
    right is mirrored across the foot's long axis at draw time only.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return np.asarray(grid) if side == "left" else np.asarray(grid)[..., ::-1]


def flat_to_grid(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat)
    grid = np.zeros((ROWS, COLS), dtype=flat.dtype)
    idx = sensor_indices()
    grid[idx[:, 0], idx[:, 1]] = flat
    return grid


def grid_to_flat(grid: np.ndarray) -> np.ndarray:
    idx = sensor_indices()
    return grid[idx[:, 0], idx[:, 1]]
