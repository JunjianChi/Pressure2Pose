"""Turn the raw capture CSV into model-ready crosstalk/reference image pairs.

Each capture logs two interleaved streams under one `client` id: client 0 is the diode
insole (a physically crosstalk-free reference, the ground truth) and client 1 is the
ordinary crossbar array (crosstalk-contaminated, the input). The two are captured under
the same load but not perfectly synchronized, so we pair each reference frame with the
nearest crosstalk frame in time and drop pairs where either side is an empty default
frame. Resistance is normalized and inverted so 1 means fully pressed, 0 means unloaded,
matching the sigmoid output of the network.
"""

from __future__ import annotations

import numpy as np

from crosstalk.sensor import R_INVALID, active_mask, flat_to_grid

MAX_DT = 0.02  # seconds; the widest gap allowed between a reference and crosstalk frame


def _read_client(df, client: int):
    """Timestamps (N,) and channel values (N, 253) for one client stream."""
    sub = df[df["client"] == client]
    t = sub["timestamp"].to_numpy(dtype=np.float64)
    vals = sub.filter(like="ch_").to_numpy(dtype=np.float32)
    return t, vals


def _is_default(frame: np.ndarray) -> np.ndarray:
    """True for rows where every channel reads the invalid fill (no sensor active)."""
    return np.all(frame >= R_INVALID, axis=1)


def pair_streams(ref_t, ref_v, ct_t, ct_v, max_dt: float = MAX_DT):
    """Return (ref_flat, ct_flat), each (N, 253), nearest-in-time and both non-empty."""
    # For each reference frame, pick the closer of the two surrounding crosstalk frames.
    after = np.searchsorted(ct_t, ref_t, side="left")
    before = after - 1
    dt_before = np.where(before >= 0, np.abs(ct_t[np.clip(before, 0, None)] - ref_t), np.inf)
    dt_after = np.where(after < len(ct_t), np.abs(ct_t[np.clip(after, None, len(ct_t) - 1)] - ref_t), np.inf)
    pick = np.where(dt_after < dt_before, after, before)
    dt = np.minimum(dt_before, dt_after)

    keep = dt <= max_dt
    ref = ref_v[keep]
    ct = ct_v[np.clip(pick[keep], 0, len(ct_t) - 1)]

    good = ~_is_default(ref) & ~_is_default(ct)
    return ref[good], ct[good]


def pair_frames(csv_path: str, max_dt: float = MAX_DT):
    """Read a capture CSV and pair its two client streams with `pair_streams`."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    ref_t, ref_v = _read_client(df, 0)
    ct_t, ct_v = _read_client(df, 1)
    if len(ref_t) == 0 or len(ct_t) == 0:
        raise ValueError("CSV is missing reference (client 0) or crosstalk (client 1) frames")
    return pair_streams(ref_t, ref_v, ct_t, ct_v, max_dt)


def pressed_from_grid(grid: np.ndarray) -> np.ndarray:
    """Resistance grid(s) (..., 33, 15) to pressed-fraction images in [0, 1].

    Low resistance means pressure, so the value is inverted; empty and faulty cells are 0.
    """
    pressed = 1.0 - np.clip(grid, 0.0, R_INVALID) / R_INVALID
    pressed = np.where(active_mask(), pressed, 0.0)
    return pressed.astype(np.float32)


def to_pressed_grid(flat: np.ndarray) -> np.ndarray:
    """Length-253 resistance vector to a (33, 15) pressed-fraction image in [0, 1]."""
    return pressed_from_grid(flat_to_grid(flat))


def prepare(csv_path: str, out_npz: str) -> dict:
    """Build paired (crosstalk, reference) pressed images from a CSV and save them."""
    ref_flat, ct_flat = pair_frames(csv_path)
    if len(ct_flat) == 0:
        raise ValueError(
            f"{csv_path} yielded no usable pairs (every frame is an empty default fill). "
            "Point --csv at a capture that contains real pressure."
        )
    ct = np.stack([to_pressed_grid(f) for f in ct_flat])   # (N, 33, 15) input
    ref = np.stack([to_pressed_grid(f) for f in ref_flat])  # (N, 33, 15) target
    np.savez(out_npz, ct=ct, ref=ref)
    return {"n_pairs": int(ct.shape[0]), "out": out_npz}


def temporal_split(n: int, val_frac: float):
    """Contiguous train/validation index split: validation is the final stretch.

    A contiguous tail separates the two in time only when the archive is in capture order,
    where adjacent frames are near-duplicates. The shipped lab archives were shuffled by an
    upstream `random_split`, so there the tail is an arbitrary fifth of the frames.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be strictly between 0 and 1")
    cut = int(n * (1.0 - val_frac))
    if cut == 0 or cut == n:
        raise ValueError(f"val_frac {val_frac} leaves an empty partition for n={n}")
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


class CrosstalkDataset:
    """Serves (input, target, mask) where input stacks the crosstalk image and the mask."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.ct = d["ct"]
        self.ref = d["ref"]
        self.mask = active_mask().astype(np.float32)

    def __len__(self):
        return len(self.ct)

    def __getitem__(self, i):
        x = np.stack([self.ct[i], self.mask])          # (2, 33, 15)
        y = self.ref[i][None]                          # (1, 33, 15)
        m = self.mask[None]                            # (1, 33, 15)
        return x.astype(np.float32), y.astype(np.float32), m
