"""What the cross-insole resampling is and is not allowed to change.

The operator is geometric and subject-independent, so these are exact identities rather than
tolerances on a fit. The one that matters most is the last: conserving the sum of pressure values is
not the same as conserving force, and the difference is a per-subject bias because MovePort ships two
insole sizes.
"""
from __future__ import annotations

import numpy as np
import pytest

from posesim.data import insole as ours
from posesim.data.register import coverage, resample_matrix, unit_coords


def _mask(rows, cols, pad=1):
    m = np.zeros((rows, cols), dtype=bool)
    m[pad:rows - pad, pad:cols - pad] = True
    return m


def test_pressure_sum_is_conserved_by_default():
    src, dst = _mask(31, 11), ours.active_mask()
    W = resample_matrix(src, dst)
    p = np.random.default_rng(0).random(int(src.sum()))
    assert (W @ p).sum() == pytest.approx(p.sum(), rel=1e-9)


def test_area_ratio_conserves_force_instead():
    """With the area ratio applied, force survives the change of array exactly.

    Force is ``sum(pressure) * cell_area``. Without the ratio the 63.72 mm2 insole loses 11.7% of
    its force on the way to our 56.25 mm2 array while the 54.34 mm2 insole gains 3.5% -- opposite
    directions, and insole size tracks foot size, so it is a confound rather than a scale.
    """
    src, dst = _mask(31, 11), ours.active_mask()
    p = np.random.default_rng(1).random(int(src.sum())) * 40.0
    for a_src in (63.72e-6, 54.34e-6):
        W = resample_matrix(src, dst, area_ratio=a_src / ours.CELL_AREA_M2)
        assert (W @ p).sum() * ours.CELL_AREA_M2 == pytest.approx(p.sum() * a_src, rel=1e-9)


def test_area_ratio_of_one_is_the_default():
    src, dst = _mask(31, 11), ours.active_mask()
    assert np.allclose(resample_matrix(src, dst), resample_matrix(src, dst, area_ratio=1.0))


def test_row_wise_coordinates_span_each_row():
    """Every row of the wired region reaches both edges, which is what a global width factor loses."""
    v = unit_coords(ours.active_mask())[:, 1]
    assert v.min() == pytest.approx(0.0)
    assert v.max() == pytest.approx(1.0)


def test_every_target_cell_is_covered_by_moveport():
    """Zero unmeasured cells -- the property row-wise normalisation was introduced to obtain."""
    assert coverage(_mask(31, 11), ours.active_mask()).all()


def test_resampling_is_non_negative():
    src, dst = _mask(31, 11), ours.active_mask()
    assert (resample_matrix(src, dst) >= 0).all()
