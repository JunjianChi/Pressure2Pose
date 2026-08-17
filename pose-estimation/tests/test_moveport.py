"""Left/right handling for MovePort, which stores its two insoles the opposite way to the 253 rig.

MovePort keeps each insole in its own physical frame, so the raw arrays are already a mirror pair:
flipping the right one makes the wired masks agree exactly (+1.0000, against -0.2457 unflipped). The
253-cell rig reads both feet through the same matrix orientation, so it needs the opposite treatment.
Applying one convention to both is what drew two identical-looking feet in the first QC figure, so
the two purposes are separate functions here and both directions are pinned.
"""
import numpy as np
import pytest

from posesim.data.moveport import COLS, ROWS, for_display, to_canonical


def asymmetric_foot():
    """A grid with a distinct medial bulge, so a mirror is detectable rather than a no-op."""
    g = np.zeros((ROWS, COLS))
    g[2:6, 7:10] = 1.0                    # a toe blob well off the centre line
    g[24:29, 4:7] = 0.6                   # a heel blob nearer the middle
    return g


def test_canonical_flips_the_right_foot_only():
    g = asymmetric_foot()
    assert np.array_equal(to_canonical(g, "left"), g)
    assert np.array_equal(to_canonical(g, "right"), g[:, ::-1])


def test_display_leaves_both_feet_in_their_own_frame():
    """Figures must show a mirrored pair; MovePort already stores one, so display is the identity."""
    g = asymmetric_foot()
    for side in ("left", "right"):
        assert np.array_equal(for_display(g, side), g)


def test_canonical_and_display_disagree_on_the_right_foot():
    """The whole point of two functions: model input and figure want opposite things."""
    g = asymmetric_foot()
    assert not np.array_equal(to_canonical(g, "right"), for_display(g, "right"))


def test_canonical_makes_a_mirrored_pair_correspond():
    """Two feet stored as mirror images must land on the same array once canonicalised."""
    left = asymmetric_foot()
    right = left[:, ::-1]                 # the same foot as its mirror-image part
    assert np.array_equal(to_canonical(left, "left"), to_canonical(right, "right"))


def test_canonical_is_an_involution_on_the_right():
    g = asymmetric_foot()
    assert np.array_equal(to_canonical(to_canonical(g, "right"), "right"), g)


@pytest.mark.parametrize("fn", [to_canonical, for_display])
def test_side_must_be_named(fn):
    with pytest.raises(ValueError):
        fn(asymmetric_foot(), "L")
