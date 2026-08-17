import numpy as np

from crosstalk.data import pressed_from_grid
from crosstalk.sensor import COLS, R_INVALID, ROWS, active_mask
from crosstalk.synthetic import load_to_resistance, make_pairs, random_load


def test_load_is_bounded_and_zero_off_the_sensors():
    pressed = random_load(np.random.default_rng(0))
    assert pressed.shape == (ROWS, COLS)
    assert pressed.min() >= 0.0 and pressed.max() <= 1.0
    assert np.all(pressed[~active_mask()] == 0.0)


def test_resistance_round_trips_through_the_pressed_conversion():
    pressed = random_load(np.random.default_rng(1))
    recovered = pressed_from_grid(load_to_resistance(pressed))
    assert np.allclose(recovered[active_mask()], pressed[active_mask()], atol=0.02)


def test_sneak_paths_lower_every_measured_resistance():
    measured, truth = make_pairs(2, seed=0)
    m = active_mask()
    assert measured.shape == truth.shape == (2, ROWS, COLS)
    assert np.all(measured[:, m] < truth[:, m])
    assert np.isfinite(measured).all()
    assert np.all(measured[:, ~m] == R_INVALID)


def test_the_pair_is_a_learnable_signal_not_a_copy():
    measured, truth = make_pairs(4, seed=0)
    ct, ref = pressed_from_grid(measured), pressed_from_grid(truth)
    m = active_mask()
    assert ((ct[:, m] - ref[:, m]) ** 2).mean() > 0.05     # the input is far from the target
    assert np.corrcoef(ct[:, m].ravel(), ref[:, m].ravel())[0, 1] > 0.3   # but carries it
