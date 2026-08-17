"""Causal dilated temporal convolutions: output t reads input up to t and no further.

Receptive field is arithmetic, not a sweep: kernel 3 over dilations 1, 2, 4, 8 in blocks of
two gives 61 frames = 1.02 s at 60 Hz, one gait cycle.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d padded only on the left, so output t never reads input after t."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, **kw):
        super().__init__(in_ch, out_ch, kernel_size, padding=0, dilation=dilation, **kw)
        self.left_pad = dilation * (kernel_size - 1)

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_pad, 0)))


class PoseTCN(nn.Module):
    """Per-frame features to per-marker mean and log-variance.

    The variance is per marker rather than per coordinate: the question it answers is which
    landmarks a shoe can see, which is a property of the landmark. With ``head`` supplied, the
    means come from that module instead of a linear layer.
    """

    def __init__(self, feat=128, n_joints=10, hidden=96, kernel=3, dilations=(1, 2, 4, 8),
                 head=None):
        super().__init__()
        layers, ch = [], feat
        for d in dilations:
            layers += [CausalConv1d(ch, hidden, kernel, dilation=d), nn.ReLU(inplace=True),
                       CausalConv1d(hidden, hidden, kernel, dilation=d), nn.ReLU(inplace=True)]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = head or nn.Conv1d(hidden, n_joints * 4, 1)
        if isinstance(self.head, nn.Conv1d):
            # Start at unit variance: the MSE warm-up gives this head no gradient,
            # so a random initialisation would meet the likelihood loss untrained.
            # Channels run (mean x, mean y, mean z, log-variance) per joint.
            nn.init.zeros_(self.head.weight[3::4])
            nn.init.zeros_(self.head.bias[3::4])
        self.n_joints = n_joints
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(dilations)

    def forward(self, z):                      # (B,T,feat)
        h = self.net(z.transpose(1, 2))
        if not isinstance(self.head, nn.Conv1d):
            return self.head(h.transpose(1, 2))
        y = self.head(h).transpose(1, 2)
        b, t, _ = y.shape
        y = y.view(b, t, self.n_joints, 4)
        return y[..., :3], y[..., 3]           # mean (B,T,J,3), log-variance (B,T,J)
