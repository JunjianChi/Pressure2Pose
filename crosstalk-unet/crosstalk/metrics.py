"""Masked regression metrics over active sensor cells.

Inactive array positions never enter a score. `MaskedScore` accumulates across
batches, so a streaming evaluation and a whole-array evaluation of the same
predictions produce identical numbers; it accepts torch tensors and numpy arrays
alike.
"""

from __future__ import annotations


def masked_mse(pred, target, mask):
    """Differentiable mean squared error over cells where mask is 1."""
    se = mask * (pred - target) ** 2
    return se.sum() / mask.sum().clamp_min(1.0)


class MaskedScore:
    """Streaming masked MSE and R^2 accumulator."""

    def __init__(self):
        self.se = 0.0
        self.n = 0.0
        self.tsum = 0.0
        self.tsq = 0.0

    def update(self, pred, target, mask):
        self.se += float((mask * (pred - target) ** 2).sum())
        self.n += float(mask.sum())
        self.tsum += float((mask * target).sum())
        self.tsq += float((mask * target * target).sum())

    @property
    def mse(self) -> float:
        return self.se / max(self.n, 1.0)

    @property
    def r2(self) -> float:
        n = max(self.n, 1.0)
        var = self.tsq / n - (self.tsum / n) ** 2
        return 1.0 - self.mse / var if var > 0 else float("nan")
