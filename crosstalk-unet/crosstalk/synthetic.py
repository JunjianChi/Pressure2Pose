"""Generate paired frames from the simulator, so the pipeline runs without the lab capture.

A random smooth load map over the active cells becomes a ground-truth resistance
frame; `CrossbarReadout` measures it with sneak paths included. The pair plays the
same roles the capture does: the measured frame is the input, the ground-truth frame
is the target. The distortion here comes from the simulator's potential-divider
excitation, not from the insole's Howland front end, so a model trained on these
pairs is a working demonstration of the pipeline and not a substitute for the lab
result.
"""

from __future__ import annotations

import numpy as np

from crosstalk.sensor import COLS, R_INVALID, ROWS, active_mask
from crosstalk.simulate import CrossbarReadout

R_PRESSED = 500.0    # ohm at full load
R_OPEN = 1e9         # ohm at an array position with no sensor
SIGMA = 2.5          # smoothing of the random load field, in cells
SPARSITY = 2.0       # exponent that pushes light contact toward zero


def random_load(rng: np.random.Generator, sigma: float = SIGMA, sparsity: float = SPARSITY) -> np.ndarray:
    """A (33, 15) pressed-fraction map in [0, 1]: smoothed noise, zero off the sensors."""
    from scipy.ndimage import gaussian_filter

    field = gaussian_filter(rng.random((ROWS, COLS)), sigma=sigma)
    mask = active_mask()
    lo, hi = field[mask].min(), field[mask].max()
    pressed = (field - lo) / (hi - lo) if hi > lo else np.zeros_like(field)
    pressed = np.clip(pressed, 0.0, 1.0) ** sparsity
    return np.where(mask, pressed, 0.0)


def load_to_resistance(pressed: np.ndarray) -> np.ndarray:
    """Pressed fraction to a resistance frame: full load at `R_PRESSED`, no load at `R_INVALID`."""
    r = R_PRESSED + (R_INVALID - R_PRESSED) * (1.0 - pressed)
    return np.where(active_mask(), r, R_OPEN)


def make_pairs(n: int, seed: int = 0, readout: CrossbarReadout | None = None, progress=None):
    """`n` (measured, ground-truth) resistance frame pairs, each (n, 33, 15) in ohms."""
    readout = readout or CrossbarReadout()
    rng = np.random.default_rng(seed)
    truth = np.empty((n, ROWS, COLS))
    measured = np.empty((n, ROWS, COLS))
    mask = active_mask()
    for i in range(n):
        truth[i] = load_to_resistance(random_load(rng))
        # An array position with no sensor reads open, which the readout returns as inf.
        # The hardware writes R_INVALID there, so the frame stays a valid resistance frame.
        frame = readout.measure(truth[i])
        measured[i] = np.where(mask & np.isfinite(frame), frame, R_INVALID)
        if progress is not None:
            progress(i + 1, n)
    return measured, truth
