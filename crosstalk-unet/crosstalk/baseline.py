"""Dense-network baseline with the same input and output shapes as `UNet`.

The training and evaluation scripts can therefore swap models directly. It is
parameter-matched to the U-Net, so the score gap between the two measures locality
and weight sharing against a dense bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crosstalk.sensor import COLS, ROWS


class FrameMLP(nn.Module):
    """Flattened crossbar frame plus mask to a corrected frame, one hidden layer."""

    def __init__(self, in_ch: int = 2, hidden: int = 300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * ROWS * COLS, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ROWS * COLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).view(-1, 1, ROWS, COLS)
