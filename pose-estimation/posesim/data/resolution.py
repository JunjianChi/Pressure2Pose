"""Active-cell block averaging for the E3 resolution counterfactual.

Each block mean is written back to every active cell of that block on the
original array, so the active-cell sum is preserved and pooling commutes
with the unit-sum shape normalisation. Inactive cells stay zero because they
never enter the flat 253-cell representation.
"""
from __future__ import annotations

import hashlib

import numpy as np

from posesim.data import insole as ours

_INDEX_CACHE: dict[tuple[str, int], np.ndarray] = {}
_ROW_CHUNK = 65536


def block_index(mask: np.ndarray, block: int, origin: int = 0) -> np.ndarray:
    """Block identifier per active cell.

    The grid starts at array (0, 0) by default; ``origin`` shifts it by that
    many cells on both axes, which is the phase-shift sensitivity check.
    """
    if not isinstance(block, (int, np.integer)) or block < 1:
        raise ValueError("block must be a positive integer")
    if not isinstance(origin, (int, np.integer)) or origin < 0:
        raise ValueError("origin must be a non-negative integer")
    mask = np.asarray(mask, dtype=bool)
    key = (hashlib.sha256(np.ascontiguousarray(mask)).hexdigest(), int(block), int(origin))
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    rows, cols = np.nonzero(mask)
    shift = int(origin) % int(block)
    row_blocks = (rows + shift) // block
    col_blocks = (cols + shift) // block
    index = row_blocks * (int(col_blocks.max()) + 1) + col_blocks
    unique, index = np.unique(index, return_inverse=True)
    _INDEX_CACHE[key] = index
    return index


def block_average(pressure: np.ndarray, block: int, mask: np.ndarray | None = None,
                  origin: int = 0) -> np.ndarray:
    """Replace every active cell with the mean of its block's active cells."""
    if not isinstance(block, (int, np.integer)) or block < 1:
        raise ValueError("block must be a positive integer")
    if not isinstance(origin, (int, np.integer)) or origin < 0:
        raise ValueError("origin must be a non-negative integer")
    values = np.asarray(pressure)
    if values.shape[-1] != ours.N_SENSORS:
        raise ValueError(f"pressure must end in {ours.N_SENSORS} sensors")
    if block == 1:
        return values.copy()
    mask = ours.active_mask() if mask is None else mask
    index = block_index(mask, block, origin)
    blocks = int(index.max()) + 1
    counts = np.bincount(index, minlength=blocks)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty_like(flat, dtype=values.dtype)
    # Rows are independent, so the chunk bound changes peak memory only: the
    # whole-array form materialises a float64 copy of every sensor at once.
    for start in range(0, len(flat), _ROW_CHUNK):
        piece = flat[start:start + _ROW_CHUNK]
        sums = np.zeros((len(piece), blocks), dtype=np.float64)
        np.add.at(sums.T, index, piece.T)
        out[start:start + _ROW_CHUNK] = (sums / np.maximum(counts, 1))[:, index]
    return out.reshape(values.shape)
