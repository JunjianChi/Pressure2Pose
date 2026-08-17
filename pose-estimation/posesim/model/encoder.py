"""Pressure-map encoders and the pose network.

Global average pooling is deliberately absent: it collapses the array, so a heel-loaded and
a forefoot-loaded footprint with similar local texture map to the same vector, and a 253-cell
map carries what a 16-cell map does. That would flatten the resolution curve for a reason
unrelated to resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from posesim.data import insole as ours
from posesim.model.kinematic_head import KinematicHead
from posesim.model.temporal import PoseTCN

MOMENTS = 6      # per foot: total, CoP u, CoP v, spread uu, spread vv, contact fraction
MOMENT_HIDDEN = 2504  # With 2 feet and feat=128, matches PressureEncoder within 8 parameters.


def unit_grid(mask):
    """Normalised (u, v) of every array position, as (2, H, W); u along the foot."""
    h, w = mask.shape
    u = torch.linspace(0.0, 1.0, h).view(h, 1).expand(h, w)
    v = torch.linspace(0.0, 1.0, w).view(1, w).expand(h, w)
    return torch.stack([u, v])


def physical_moments(x, coords, eps=1e-6):
    """Total load, centre of pressure and its spread, per channel, as (B, C*MOMENTS).

    Exactly computable and known-informative, so they are concatenated rather than left for
    the encoder to rediscover through a pooling layer that may discard them.
    """
    b, c = x.shape[:2]
    flat = x.flatten(2)                                     # (B,C,HW)
    total = flat.sum(-1)
    p = flat / (total.unsqueeze(-1) + eps)
    u, v = coords.flatten(1)                                # (HW,)
    cu, cv = (p * u).sum(-1), (p * v).sum(-1)
    su = (p * (u - cu.unsqueeze(-1)) ** 2).sum(-1)
    sv = (p * (v - cv.unsqueeze(-1)) ** 2).sum(-1)
    area = (flat > 0).float().mean(-1)
    return torch.stack([total, cu, cv, su, sv, area], dim=-1).view(b, c * MOMENTS)


class PressureEncoder(nn.Module):
    """Footprint image to a per-frame vector, position-aware and pooling-free.

    Coordinate channels are appended so shared-weight convolution can be position-dependent
    (worth 5.8 cm of 19.8 in Intelligent Carpet's ablation), which matters here because
    registration has already made cell (i, j) mean the same anatomy on every subject.
    """

    def __init__(self, in_ch=2, feat=128, hidden=32, mask=None):
        super().__init__()
        m = ours.active_mask() if mask is None else mask
        self.register_buffer("mask", torch.as_tensor(m, dtype=torch.float32))
        self.register_buffer("coords", unit_grid(self.mask))
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + 2, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        h, w = self.mask.shape
        gh, gw = -(-h // 4), -(-w // 4)         # two stride-2 stages
        self.fc = nn.Linear(hidden * 2 * gh * gw + in_ch * MOMENTS, feat)

    def forward(self, x):                                   # (B,T,C,H,W)
        b, t = x.shape[:2]
        x = x.flatten(0, 1) * self.mask                     # the network need not learn the outline
        coords = self.coords.expand(x.shape[0], -1, -1, -1)
        z = self.conv(torch.cat([x, coords], 1)).flatten(1)
        z = torch.cat([z, physical_moments(x, self.coords)], -1)
        return self.fc(z).view(b, t, -1)


class MomentEncoder(nn.Module):
    """The null hypothesis: a footprint reduced to its low-order moments.

    Deployed insole systems typically feed force, centre of pressure, and contact
    summaries rather than a map. The MLP is widened to match the dense encoder's
    parameter budget, so E1 isolates the representation.
    """

    def __init__(self, in_ch=2, feat=128, hidden=MOMENT_HIDDEN, mask=None):
        super().__init__()
        m = ours.active_mask() if mask is None else mask
        self.register_buffer("mask", torch.as_tensor(m, dtype=torch.float32))
        self.register_buffer("coords", unit_grid(self.mask))
        self.fc = nn.Sequential(nn.Linear(in_ch * MOMENTS, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, feat))

    def forward(self, x):
        b, t = x.shape[:2]
        z = physical_moments(x.flatten(0, 1) * self.mask, self.coords)
        return self.fc(z).view(b, t, -1)


class PosePressureNet(nn.Module):
    """Pressure (and optionally IMU) to lower-body marker positions.

    ``head="free"`` regresses the ten markers directly; ``head="kinematic"`` regresses SMPL
    joint rotations and a shape vector, which makes bone lengths exact instead of penalised.
    ``encoder="none"`` drops the pressure branch for the inertial-only variant.
    """

    def __init__(self, in_ch=2, feat=128, n_joints=10, imu_dim=0, encoder="dense",
                 head="free", hidden=96, dilations=(1, 2, 4, 8), moment_hidden=MOMENT_HIDDEN,
                 **kw):
        super().__init__()
        if encoder == "none":
            if not imu_dim:
                raise ValueError("the inertial-only encoder needs an inertial input")
            self.enc = None
            feat = 0
        elif encoder == "moments":
            self.enc = MomentEncoder(in_ch, feat, hidden=moment_hidden)
        else:
            self.enc = PressureEncoder(in_ch, feat)
        self.imu_dim = imu_dim
        out = KinematicHead(hidden, **kw) if head == "kinematic" else None
        self.tcn = PoseTCN(feat + imu_dim, n_joints, hidden=hidden, head=out,
                           dilations=tuple(dilations))

    def forward(self, x, imu=None):
        if self.imu_dim and imu is None:
            raise ValueError("model built with imu_dim > 0 but no imu passed to forward")
        if self.enc is None:
            return self.tcn(imu)
        z = self.enc(x)
        if self.imu_dim:
            z = torch.cat([z, imu], dim=-1)
        return self.tcn(z)
