"""SMPL joint rotations as the output, supervised in marker space.

Bone lengths come from SMPL's shape-dependent joint template, so a limb cannot change length.
The label stays the measured marker, because the head is a differentiable parameterisation of
the output and not a fitted target. Only the joints on the two leg chains are predicted; the
6890-vertex mesh never enters, so a marker is a rigid offset in its parent joint's frame.

Rotation representation: Zhou et al., "On the Continuity of Rotation Representations in Neural
Networks", CVPR 2019.

The joint template it loads comes from SMPL, which carries a non-commercial licence from
its publisher; this code is Apache-2.0 but that file is not.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# SMPL ids on the two leg chains, in local order.
SMPL_IDS = (0, 1, 4, 7, 2, 5, 8)                 # pelvis, L hip/knee/ankle, R hip/knee/ankle
PARENTS = (-1, 0, 1, 2, 0, 4, 5)
L_KNEE, L_ANKLE, R_KNEE, R_ANKLE = 2, 3, 5, 6

# Which joint each of the ten markers rides on. The epicondyles ride the knee; the malleolus,
# calcaneus and metatarsal head all ride the foot segment, whose proximal joint is the ankle.
MARKER_PARENT = (L_KNEE, R_KNEE, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
                 L_ANKLE, R_ANKLE, L_ANKLE, R_ANKLE)


def rot6d_to_matrix(x):
    """(..., 6) to (..., 3, 3) by Gram-Schmidt on the first two columns."""
    a, b = x[..., :3], x[..., 3:]
    e1 = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], dim=-1)


def causal_mean(x, dim=1):
    """Running mean over ``dim``, so frame t averages only frames up to t."""
    n = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.dim()
    shape[dim] = -1
    return x.cumsum(dim) / n.view(shape)


class KinematicHead(nn.Module):
    """Per-frame features to marker positions, through joint rotations and a shape vector.

    Beta is averaged causally across the window: it is a property of the subject, so letting it
    move frame to frame would reintroduce the varying bone lengths this head exists to forbid.
    """

    def __init__(self, feat, joints_npz="data/processed/smpl_joints.npz", n_betas=10):
        super().__init__()
        d = np.load(joints_npz)
        ids = list(SMPL_IDS)
        self.register_buffer("j_rest", torch.as_tensor(d["J_rest"][ids]))            # (7,3)
        self.register_buffer("j_beta", torch.as_tensor(d["J_beta"][ids, :, :n_betas]))
        self.register_buffer("parents", torch.as_tensor(np.array(PARENTS)))
        self.register_buffer("marker_parent", torch.as_tensor(np.array(MARKER_PARENT)))
        self.offset = nn.Parameter(torch.zeros(len(MARKER_PARENT), 3))
        self.rot = nn.Linear(feat, len(SMPL_IDS) * 6)
        self.beta = nn.Linear(feat, n_betas)
        self.logvar = nn.Linear(feat, len(MARKER_PARENT))

    def joints(self, betas):
        """Rest-pose joint positions for a shape, as (..., 7, 3)."""
        return self.j_rest + torch.einsum("jdb,...b->...jd", self.j_beta, betas)

    def forward(self, z):                                    # (B,T,feat)
        b, t, _ = z.shape
        R = rot6d_to_matrix(self.rot(z).view(b, t, len(SMPL_IDS), 6))
        J = self.joints(causal_mean(self.beta(z)))           # (B,T,7,3)

        Rg = [R[:, :, 0]]
        tg = [torch.zeros(b, t, 3, device=z.device, dtype=z.dtype)]   # pelvis frame origin
        for j in range(1, len(SMPL_IDS)):
            p = PARENTS[j]
            Rg.append(Rg[p] @ R[:, :, j])
            tg.append(tg[p] + torch.einsum("btij,btj->bti", Rg[p], J[:, :, j] - J[:, :, p]))
        Rg, tg = torch.stack(Rg, 2), torch.stack(tg, 2)       # (B,T,7,3,3), (B,T,7,3)

        par = self.marker_parent
        markers = tg[:, :, par] + torch.einsum("btmij,mj->btmi", Rg[:, :, par], self.offset)
        return markers, self.logvar(z)

    def bone_lengths(self, z):
        """Segment lengths implied by the predicted shape, as (B, T, 6) in metres."""
        J = self.joints(causal_mean(self.beta(z)))
        p = torch.as_tensor(PARENTS[1:], device=z.device)
        return (J[:, :, 1:] - J[:, :, p]).norm(dim=-1)
