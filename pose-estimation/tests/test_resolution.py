"""Admission tests for the E3 block-average resolution operator."""
from __future__ import annotations

import numpy as np
import pytest

from posesim.data import insole as ours
from posesim.data.resolution import block_average, block_index


def test_block_one_is_the_identity():
    rng = np.random.default_rng(0)
    pressure = rng.random((5, 2, ours.N_SENSORS)).astype(np.float32)
    assert np.allclose(block_average(pressure, 1), pressure)


@pytest.mark.parametrize("block", (2, 3, 4, 6, 8))
def test_block_average_preserves_the_active_cell_sum(block):
    rng = np.random.default_rng(1)
    pressure = rng.random((7, 2, ours.N_SENSORS)).astype(np.float64)
    pooled = block_average(pressure, block)
    assert pooled.shape == pressure.shape
    assert np.allclose(pooled.sum(axis=-1), pressure.sum(axis=-1))


def test_cells_inside_one_block_share_the_block_mean():
    mask = ours.active_mask()
    index = block_index(mask, 4)
    rng = np.random.default_rng(2)
    pressure = rng.random((1, 1, ours.N_SENSORS))
    pooled = block_average(pressure, 4)
    for block in np.unique(index):
        members = index == block
        assert np.allclose(pooled[0, 0, members], pooled[0, 0, members][0])
        assert np.isclose(pooled[0, 0, members][0], pressure[0, 0, members].mean())


def test_block_grid_originates_at_array_zero_and_covers_edges():
    mask = ours.active_mask()
    index = block_index(mask, 6)
    rows, cols = np.nonzero(mask)
    # Two active cells share a block exactly when they share a 6x6 tile of the
    # array measured from (0, 0); the label values themselves are arbitrary.
    tile = list(zip(rows // 6, cols // 6))
    for a in range(0, len(index), 7):
        for b in range(0, len(index), 11):
            assert (index[a] == index[b]) == (tile[a] == tile[b])
    assert index.min() == 0
    assert len(np.unique(index)) > 1


def test_coarser_blocks_lose_spatial_contrast():
    rng = np.random.default_rng(3)
    pressure = rng.random((4, 2, ours.N_SENSORS))
    spreads = [float(block_average(pressure, b).std(axis=-1).mean()) for b in (1, 2, 4, 8)]
    assert spreads == sorted(spreads, reverse=True)


def test_index_cache_distinguishes_different_masks_of_one_shape():
    first = ours.active_mask()
    second = first.copy()
    moved = np.flatnonzero(second.ravel())[0]
    free = np.flatnonzero(~second.ravel())[0]
    second.ravel()[moved], second.ravel()[free] = False, True
    assert not np.array_equal(block_index(first, 4), block_index(second, 4))


def test_phase_shifted_grid_changes_the_pooling_but_keeps_the_sum():
    rng = np.random.default_rng(4)
    pressure = rng.random((3, 2, ours.N_SENSORS))
    aligned = block_average(pressure, 4)
    shifted = block_average(pressure, 4, origin=2)
    assert not np.allclose(aligned, shifted)
    assert np.allclose(shifted.sum(axis=-1), pressure.sum(axis=-1))
    assert np.allclose(block_average(pressure, 4, origin=0), aligned)
    assert np.allclose(block_average(pressure, 4, origin=4), aligned)   # one full period
    with pytest.raises(ValueError):
        block_average(pressure, 4, origin=-1)


def test_rejects_invalid_block_sizes():
    pressure = np.zeros((2, 2, ours.N_SENSORS))
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            block_average(pressure, bad)


def test_the_row_chunk_bound_changes_memory_not_values():
    """Rows are independent; a chunk boundary must not move a single value."""
    from posesim.data import resolution

    rng = np.random.default_rng(0)
    pressure = rng.random((resolution._ROW_CHUNK + 37, ours.N_SENSORS)) * 1e4
    reference = resolution.block_average(pressure, 3)
    original = resolution._ROW_CHUNK
    try:
        resolution._ROW_CHUNK = 101
        chunked = resolution.block_average(pressure, 3)
    finally:
        resolution._ROW_CHUNK = original
    assert np.array_equal(reference, chunked)
    assert reference.shape == pressure.shape
