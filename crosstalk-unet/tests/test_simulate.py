import numpy as np
import pytest

from crosstalk.simulate import CrossbarReadout


def test_single_cell_has_no_sneak_path():
    readout = CrossbarReadout()
    measured = readout.measure(np.array([[1234.0]]))
    assert measured[0, 0] == pytest.approx(1234.0, rel=1e-9)


def test_open_neighbours_recover_the_true_value():
    r = np.full((4, 3), 1e12)
    r[1, 1] = 2000.0
    measured = CrossbarReadout().measure_cell(r, 1, 1)
    assert measured == pytest.approx(2000.0, rel=1e-6)


def test_sneak_paths_only_lower_the_reading():
    r = np.full((4, 3), 5000.0)
    measured = CrossbarReadout().measure(r)
    assert (measured < 5000.0).all()


def test_uniform_grid_measures_uniformly():
    measured = CrossbarReadout().measure(np.full((5, 4), 3000.0))
    assert np.allclose(measured, measured[0, 0])


def test_pressed_cluster_contaminates_an_unpressed_neighbour():
    background = 20000.0
    r = np.full((6, 5), background)
    r[0:2, 0:2] = 200.0  # a firmly pressed cluster
    clean = CrossbarReadout().measure_cell(np.full((6, 5), background), 0, 4)
    contaminated = CrossbarReadout().measure_cell(r, 0, 4)
    assert contaminated < clean  # the cluster's sneak paths drag the far cell down


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        CrossbarReadout().measure(np.array([[0.0]]))
    with pytest.raises(ValueError):
        CrossbarReadout(r_series=0.0)
