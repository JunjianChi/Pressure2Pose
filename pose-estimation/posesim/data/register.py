"""Resample footprints between insole arrays via a shared unit-foot coordinate.

MovePort's 31x11 (230 wired) and our 33x15 (253 wired) differ in grid, pitch and outline.
Reference for outline-to-outline registration: Pataky, Gait & Posture 29(3), 2009.
"""
from __future__ import annotations

import numpy as np


def array_extent(mask):
    """Row and column bounds of the wired region, as (r0, r1, c0, c1) inclusive."""
    rr, cc = np.nonzero(mask)
    return int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())


def unit_coords(mask, per_row=True):
    """Wired cell centres in [0, 1]^2: u along the foot, v across it.

    ``per_row`` scales v by the width of its own row rather than of the whole outline, so
    the two insoles correspond along their full length instead of only at the widest point.
    """
    r0, r1, c0, c1 = array_extent(mask)
    rr, cc = np.nonzero(mask)
    u = (rr - r0) / max(r1 - r0, 1)
    if not per_row:
        return np.stack([u, (cc - c0) / max(c1 - c0, 1)], axis=1)
    v = np.empty(len(rr), dtype=float)
    for r in np.unique(rr):
        sel = rr == r
        lo, hi = cc[sel].min(), cc[sel].max()
        v[sel] = (cc[sel] - lo) / max(hi - lo, 1) if hi > lo else 0.5
    return np.stack([u, v], axis=1)


def resample_matrix(src_mask, dst_mask, cells=0.25, area_ratio=None):
    """Weights W with ``dst = W @ src``, spreading each source cell by a Gaussian.

    ``cells`` is the width in units of a *target* cell, applied separately along and across the
    foot; a single sigma in unit-foot coordinates is three times wider across than along. Weights
    are normalised per source cell, which conserves the sum of pressure values. Pass
    ``area_ratio = src_cell_area / dst_cell_area`` to conserve force instead.

    The default is 0.25 because the two arrays are nearly the same granularity -- 31 rows against
    33, median row width 7 against 7 -- so this is a re-indexing rather than an upsampling and a
    wide kernel only destroys contrast. Measured on loaded frames, forward and inverse matched:

    ======  =========  ==========  =========
    sigma   peak kept  round-trip  var kept
    ======  =========  ==========  =========
    0.25    0.895      0.970       0.860
    0.35    0.845      0.982       0.824
    0.5     0.774      0.978       0.785
    ======  =========  ==========  =========

    0.5 is dominated: it loses 23% of the peak and 21% of the spatial contrast to buy 0.008 of
    round-trip. For a project whose subject is spatial resolution, blurring the input before the
    model sees it destroys the quantity being measured.
    """
    su, sv = unit_coords(src_mask).T
    du, dv = unit_coords(dst_mask).T
    r0, r1, c0, c1 = array_extent(dst_mask)
    d2 = (((du[:, None] - su[None, :]) / (cells / max(r1 - r0, 1))) ** 2
          + ((dv[:, None] - sv[None, :]) / (cells / max(c1 - c0, 1))) ** 2)
    w = np.exp(-0.5 * d2)
    w /= np.clip(w.sum(axis=0, keepdims=True), 1e-12, None)
    return w if area_ratio is None else w * float(area_ratio)


def coverage(src_mask, dst_mask, max_cells=1.2):
    """Target cells with a source cell within ``max_cells``.

    Distance to the nearest source, not the weight sum: a cell filled entirely by distant
    sources received load without measuring it.
    """
    su, sv = unit_coords(src_mask).T
    du, dv = unit_coords(dst_mask).T
    r0, r1, c0, c1 = array_extent(dst_mask)
    d = np.sqrt((((du[:, None] - su[None, :]) * max(r1 - r0, 1)) ** 2)
                + (((dv[:, None] - sv[None, :]) * max(c1 - c0, 1)) ** 2))
    return d.min(axis=1) <= max_cells


def grid_from_flat(flat, mask):
    """Scatter flat wired values back into their (rows, cols) image, NaN elsewhere."""
    out = np.full(mask.shape, np.nan)
    out[mask] = flat
    return out
