"""Model invariants: causality, shapes, and the two encoders' parameter budgets."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from posesim.model.encoder import MomentEncoder, PosePressureNet, PressureEncoder, physical_moments
from posesim.model.temporal import PoseTCN


_SMPL_JOINTS = Path("data/processed/smpl_joints.npz")
requires_smpl_joints = pytest.mark.skipif(
    not _SMPL_JOINTS.is_file(), reason="requires untracked data/processed/smpl_joints.npz"
)


def _batch(b=2, t=64):
    return torch.rand(b, t, 2, 33, 15), torch.randn(b, t, 12)


def test_output_at_t_ignores_everything_after_t():
    """SIG-1. A centred kernel passes this only by accident, so it is asserted, not assumed."""
    net = PoseTCN(feat=16, n_joints=10, hidden=16).eval()
    z = torch.randn(1, 64, 16)
    with torch.no_grad():
        a, _ = net(z)
        z2 = z.clone()
        z2[:, 40:] = torch.randn(1, 24, 16)
        b, _ = net(z2)
    assert torch.equal(a[:, :40], b[:, :40])


def test_receptive_field_is_one_gait_cycle():
    assert PoseTCN().receptive_field == 61          # 1.02 s at 60 Hz


def test_both_encoders_produce_the_same_output_shape():
    x, imu = _batch()
    for encoder in ("dense", "moments"):
        mu, logvar = PosePressureNet(encoder=encoder, imu_dim=12)(x, imu)
        assert mu.shape == (2, 64, 10, 3)
        assert logvar.shape == (2, 64, 10)


def test_parameter_budget():
    """370 independent segments does not support a model of this literature's usual size."""
    for encoder, ceiling in (("dense", 1.0e6), ("moments", 1.0e6)):
        n = sum(p.numel() for p in PosePressureNet(encoder=encoder, imu_dim=12).parameters())
        assert n < ceiling, f"{encoder} has {n} parameters"


def test_moment_encoder_matches_dense_parameter_count():
    """E1 must not turn a capacity difference into an apparent spatial-information gain."""
    dense = sum(p.numel() for p in PosePressureNet(encoder="dense", imu_dim=14).parameters())
    moments = sum(p.numel() for p in PosePressureNet(encoder="moments", imu_dim=14).parameters())
    assert abs(dense - moments) <= 16


def test_moments_are_translation_sensitive():
    """The property global average pooling destroys: where the load sits must change the code."""
    enc = PressureEncoder()
    heel, toe = torch.zeros(1, 1, 2, 33, 15), torch.zeros(1, 1, 2, 33, 15)
    heel[..., 28, 7] = 1.0
    toe[..., 3, 7] = 1.0
    assert not torch.allclose(enc(heel), enc(toe))


def test_physical_moments_track_the_centre_of_pressure():
    coords = PressureEncoder().coords
    x = torch.zeros(1, 1, 33, 15)
    x[0, 0, 8, 7] = 1.0
    near = physical_moments(x, coords)[0, 1]
    x = torch.zeros(1, 1, 33, 15)
    x[0, 0, 24, 7] = 1.0
    far = physical_moments(x, coords)[0, 1]
    assert far > near


def test_imu_is_required_when_the_model_was_built_with_it():
    x, _ = _batch()
    try:
        PosePressureNet(imu_dim=12)(x)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_moment_encoder_ignores_structure_beyond_its_moments():
    """The null hypothesis must actually be blind to layout, or E1 measures nothing."""
    enc = MomentEncoder()
    a, b = torch.zeros(1, 1, 2, 33, 15), torch.zeros(1, 1, 2, 33, 15)
    a[..., 16, 6] = a[..., 16, 8] = 1.0
    b[..., 16, 7] = 2.0
    assert torch.allclose(physical_moments(a.flatten(0, 1), enc.coords)[:, :3],
                          physical_moments(b.flatten(0, 1), enc.coords)[:, :3], atol=1e-5)


def test_link_variance_is_zero_for_a_rigid_window():
    from posesim.model.losses import link_variance
    y = torch.randn(2, 1, 10, 3).expand(2, 64, 10, 3)
    assert float(link_variance(y, links=((0, 1),))) < 1e-6


def test_link_variance_grows_when_a_segment_stretches():
    from posesim.model.losses import link_variance
    y = torch.randn(2, 64, 10, 3)
    stretched = y.clone()
    stretched[:, :, 8] = stretched[:, :, 6] + (stretched[:, :, 8] - stretched[:, :, 6]) \
        * torch.linspace(0.5, 1.5, 64).view(1, 64, 1)
    links = ((6, 8),)
    assert float(link_variance(stretched, links=links)) > float(link_variance(y, links=links))


def test_link_variance_has_no_unvalidated_default_links():
    from posesim.model.losses import RIGID_LINKS, link_variance

    y = torch.randn(2, 4, 10, 3)
    assert RIGID_LINKS == ()
    assert float(link_variance(y)) == 0.0


@requires_smpl_joints
def test_kinematic_head_holds_bone_lengths_constant():
    """The property the free head can only be penalised towards; here it is construction."""
    from posesim.model.encoder import PosePressureNet
    net = PosePressureNet(imu_dim=14, head="kinematic").eval()
    x, imu = torch.rand(2, 64, 2, 33, 15), torch.randn(2, 64, 14)
    with torch.no_grad():
        z = net.tcn.net(torch.cat([net.enc(x), imu], -1).transpose(1, 2)).transpose(1, 2)
        lengths = net.tcn.head.bone_lengths(z)
    assert float((lengths.std(1) / lengths.mean(1)).max()) < 1e-3


@requires_smpl_joints
def test_kinematic_head_matches_the_smpl_template_at_zero_shape():
    from posesim.model.kinematic_head import KinematicHead
    head = KinematicHead(96)
    j = head.joints(torch.zeros(10))
    assert abs(float((j[1] - j[2]).norm()) - 0.3768) < 1e-3      # hip to knee
    assert abs(float((j[2] - j[3]).norm()) - 0.4006) < 1e-3      # knee to ankle


def test_rot6d_is_a_rotation():
    from posesim.model.kinematic_head import rot6d_to_matrix
    R = rot6d_to_matrix(torch.randn(5, 6))
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(5, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(5), atol=1e-5)


def test_causal_mean_ignores_the_future():
    from posesim.model.kinematic_head import causal_mean
    x = torch.randn(1, 64, 3)
    a = causal_mean(x)
    x2 = x.clone()
    x2[:, 40:] = torch.randn(1, 24, 3)
    assert torch.allclose(a[:, :40], causal_mean(x2)[:, :40])


def test_beta_nll_stays_finite_under_an_untrained_variance_head():
    from posesim.model.losses import beta_nll, gaussian_nll

    pred = torch.zeros(2, 4, 10, 3)
    target = torch.ones(2, 4, 10, 3)
    for logvar in (torch.full((2, 4, 10), 200.0), torch.full((2, 4, 10), -200.0)):
        for loss in (beta_nll(pred, logvar, target), gaussian_nll(pred, logvar, target)):
            assert torch.isfinite(loss), logvar[0, 0, 0].item()
    graded = torch.full((2, 4, 10), 40.0, requires_grad=True)
    beta_nll(pred, graded, target).backward()
    assert torch.isfinite(graded.grad).all()


def test_variance_head_starts_at_unit_variance():
    from posesim.model.encoder import PosePressureNet

    torch.manual_seed(0)
    model = PosePressureNet(encoder="moments", head="free", imu_dim=0, n_joints=10)
    model.eval()
    with torch.no_grad():
        _, logvar = model(torch.rand(2, 8, 2, 33, 15))
    assert torch.allclose(logvar, torch.zeros_like(logvar))


def test_inertial_only_model_ignores_pressure_entirely():
    from posesim.model.encoder import PosePressureNet

    torch.manual_seed(0)
    model = PosePressureNet(encoder="none", head="free", imu_dim=12, n_joints=10)
    model.eval()
    imu = torch.rand(2, 16, 12)
    with torch.no_grad():
        first, _ = model(torch.rand(2, 16, 2, 33, 15), imu)
        second, _ = model(torch.rand(2, 16, 2, 33, 15) * 7.0, imu)
    assert first.shape == (2, 16, 10, 3)
    assert torch.equal(first, second)
    with pytest.raises(ValueError):
        PosePressureNet(encoder="none", head="free", imu_dim=0, n_joints=10)


def test_dilations_widen_the_receptive_field():
    from posesim.model.encoder import PosePressureNet

    default = PosePressureNet(encoder="moments", head="free", imu_dim=0, n_joints=10)
    assert default.tcn.receptive_field == 61
    wide = PosePressureNet(encoder="moments", head="free", imu_dim=0, n_joints=10,
                           dilations=(1, 2, 4, 8, 16))
    assert wide.tcn.receptive_field == 125
    with torch.no_grad():
        mu, logvar = wide(torch.rand(1, 130, 2, 33, 15))
    assert mu.shape == (1, 130, 10, 3)


def test_mirroring_is_exact():
    """Left-right reflection is a valid sample, not an approximation: handedness is canonical
    upstream, so the mirror is a foot swap plus a sign flip on the lateral axis."""
    import numpy as np
    import sys, os
    from scripts.train_moveport import mirror, MIRROR_MARKERS, LATERAL

    rng = np.random.default_rng(0)
    p, imu, y = rng.random((8, 2, 253)), rng.normal(size=(8, 2, 6)), rng.normal(size=(8, 10, 3))
    pm, im, ym = mirror(p, imu, y)
    assert np.array_equal(pm, p[:, ::-1])
    assert np.allclose(ym[:, :, LATERAL], -y[:, MIRROR_MARKERS, LATERAL])
    assert np.allclose(ym[:, :, 1:], y[:, MIRROR_MARKERS, 1:])
    assert np.array_equal(mirror(*mirror(p, imu, y))[2], y)      # an involution
