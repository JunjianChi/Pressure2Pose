import numpy as np

from crosstalk.data import pair_streams
from crosstalk.sensor import R_INVALID


def _frames(*values):
    return np.stack([np.full(253, v, dtype=np.float32) for v in values])


def test_pairs_pick_the_nearest_crosstalk_frame():
    ref_t = np.array([0.0, 1.0])
    ct_t = np.array([0.005, 0.9, 0.996])
    ref, ct = pair_streams(ref_t, _frames(10.0, 20.0), ct_t, _frames(1.0, 2.0, 3.0))
    assert len(ref) == 2
    assert ct[0, 0] == 1.0  # 0.005 beats nothing on the left
    assert ct[1, 0] == 3.0  # 0.996 beats 0.9


def test_pairs_beyond_max_dt_are_dropped():
    ref_t = np.array([0.0, 5.0])
    ct_t = np.array([0.005])
    ref, ct = pair_streams(ref_t, _frames(10.0, 20.0), ct_t, _frames(1.0))
    assert len(ref) == 1
    assert ref[0, 0] == 10.0


def test_default_fill_frames_are_dropped():
    ref_t = np.array([0.0, 1.0])
    ct_t = np.array([0.001, 1.001])
    ref_v = _frames(10.0, R_INVALID)  # the second reference frame is an empty default
    ref, ct = pair_streams(ref_t, ref_v, ct_t, _frames(1.0, 2.0))
    assert len(ref) == 1
    assert ref[0, 0] == 10.0
