import numpy as np
import pytest

from crosstalk.data import temporal_split


def test_validation_is_the_contiguous_tail():
    train, val = temporal_split(100, 0.2)
    assert list(train) == list(range(80))
    assert list(val) == list(range(80, 100))


def test_partitions_are_disjoint_and_complete():
    train, val = temporal_split(37, 0.25)
    joined = np.concatenate([train, val])
    assert len(np.intersect1d(train, val)) == 0
    assert sorted(joined) == list(range(37))


def test_degenerate_fractions_are_rejected():
    with pytest.raises(ValueError):
        temporal_split(10, 0.0)
    with pytest.raises(ValueError):
        temporal_split(10, 1.0)
    with pytest.raises(ValueError):
        temporal_split(2, 0.9)  # cut = 0: no training frames left
