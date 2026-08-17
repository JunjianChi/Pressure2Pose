import os, sys
import numpy as np
from posesim.data.insole import ROWS, COLS, N_SENSORS, active_mask, sensor_indices, flat_to_grid, grid_to_flat


def test_mask_and_roundtrip():
    assert (ROWS, COLS, N_SENSORS) == (33, 15, 253)
    assert active_mask().sum() == 253
    assert sensor_indices().shape == (253, 2)
    flat = np.arange(253, dtype=float)
    assert np.allclose(grid_to_flat(flat_to_grid(flat)), flat)


def test_a_drawn_insole_mirrors_its_outline_with_its_values():
    """VIZ-2: the pair is drawn mirrored, outline included.

    The outline is asymmetric, so mirroring only the values draws one foot's
    pressure inside the other foot's shape.
    """
    import numpy as np

    from posesim.data import insole as ours
    from scripts.viz_prediction_gif import draw_pressure

    class Recorder:
        def __init__(self):
            self.images = []

        def imshow(self, image, **kwargs):
            self.images.append(np.asarray(image, dtype=float))

        def set_axis_off(self):
            pass

    row = np.linspace(1.0, 2.0, int(ours.active_mask().sum()))
    drawn = {}
    for side in ("left", "right"):
        axis = Recorder()
        draw_pressure(axis, row, side, vmax=2.0)
        drawn[side] = axis.images

    assert len(drawn["left"]) == len(drawn["right"]) == 2
    for left, right in zip(drawn["left"], drawn["right"]):
        flipped = right[..., ::-1]
        assert np.allclose(left, flipped, equal_nan=True)
        # and the flip is not a no-op, or the assertion above proves nothing
        assert not np.allclose(left, right, equal_nan=True)
