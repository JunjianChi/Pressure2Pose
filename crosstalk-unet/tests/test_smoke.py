import numpy as np
import torch

from crosstalk.data import to_pressed_grid
from crosstalk.model import UNet
from crosstalk.sensor import R_INVALID, ROWS, COLS, active_mask, flat_to_grid, sensor_indices


def test_model_forward_shape_and_range():
    x = torch.rand(2, 2, ROWS, COLS)
    y = UNet()(x)
    assert y.shape == (2, 1, ROWS, COLS)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_mask_counts():
    assert sensor_indices().shape == (253, 2)
    assert active_mask().sum() == 253 - 4  # four faulty cells removed


def test_flat_to_grid_places_all_sensors():
    grid = flat_to_grid(np.ones(253))
    assert grid.sum() == 253
    assert grid.shape == (ROWS, COLS)


def test_pressed_grid_extremes():
    # An open cell (max resistance) is unloaded -> 0; zero resistance -> fully pressed.
    unloaded = to_pressed_grid(np.full(253, R_INVALID))
    assert np.allclose(unloaded, 0.0)
    pressed = to_pressed_grid(np.zeros(253))
    assert np.allclose(pressed[active_mask()], 1.0)
    assert np.allclose(pressed[~active_mask()], 0.0)
