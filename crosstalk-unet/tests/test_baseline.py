import numpy as np
import torch

from crosstalk.baseline import FrameMLP
from crosstalk.data import pressed_from_grid, to_pressed_grid
from crosstalk.sensor import COLS, R_INVALID, ROWS, active_mask


def test_mlp_matches_the_unet_contract():
    y = FrameMLP()(torch.rand(3, 2, ROWS, COLS))
    assert y.shape == (3, 1, ROWS, COLS)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_pressed_from_grid_handles_batches_and_matches_flat_route():
    rng = np.random.default_rng(0)
    flat = rng.uniform(0.0, R_INVALID, 253).astype(np.float32)
    from crosstalk.sensor import flat_to_grid

    single = pressed_from_grid(flat_to_grid(flat))
    assert np.array_equal(single, to_pressed_grid(flat))

    batch = np.stack([flat_to_grid(flat)] * 4)
    batched = pressed_from_grid(batch)
    assert batched.shape == (4, ROWS, COLS)
    assert np.array_equal(batched[2], single)
    assert (batched[:, ~active_mask()] == 0.0).all()
