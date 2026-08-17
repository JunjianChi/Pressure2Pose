"""Model-input preprocessing shared by the training runner and the evaluator."""
from __future__ import annotations

import numpy as np
import torch


def normalise(pressure, force):
    """Shape against the two-foot total, so each foot keeps its load share and a
    swing foot stays near zero instead of amplifying a lone surviving cell.

    Pressure is clipped at zero first: registration passes the source's small
    negative noise through, and a near-cancelling swing-foot frame would
    otherwise blow up the downstream moment normalisation."""
    pressure = np.clip(pressure, 0.0, None)
    total = pressure.sum((-2, -1), keepdims=True)
    return pressure / np.clip(total, 1e-9, None), np.log1p(np.clip(force, 0, None))


def to_grid(flat, mask):
    """(..., 253) back onto the (..., 33, 15) array the encoder convolves over."""
    out = torch.zeros(flat.shape[:-1] + mask.shape, dtype=flat.dtype, device=flat.device)
    out[..., torch.as_tensor(mask)] = flat
    return out
