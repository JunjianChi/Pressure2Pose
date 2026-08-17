"""Marker losses.

Plain Gaussian NLL scales the mean's gradient by ``exp(-log sigma^2)``, so a region that fits
early shrinks its variance and takes a larger share of the gradient, and hard regions are never
recovered. beta-NLL stops that gradient in the weight; beta = 0.5 is the published default.
Train with beta-NLL, report plain NLL -- the weighted value does not reflect model quality.
"""
from __future__ import annotations


def mpjpe(pred, target):
    """Mean Euclidean distance per marker, in metres."""
    return (pred - target).norm(dim=-1).mean()


LOGVAR_LIMIT = 10.0     # exp(10) overflows nothing in float32 and bounds sq/var


def beta_nll(pred, logvar, target, beta=0.5, min_var=1e-6):
    """Heteroscedastic Gaussian NLL with the beta stop-gradient weight.

    ``beta = 0`` is plain NLL; ``beta = 1`` makes the mean's gradient exactly that of MSE.
    The log-variance is bounded before exponentiation: an MSE warm-up leaves this
    head untrained, and an unbounded exponential of it overflows on the first
    likelihood step.
    """
    var = logvar.clamp(-LOGVAR_LIMIT, LOGVAR_LIMIT).exp().clamp_min(min_var)
    sq = ((pred - target) ** 2).sum(-1)
    nll = 0.5 * (sq / var + var.log() * pred.shape[-1])
    return (nll * var.detach() ** beta).mean() if beta else nll.mean()


def gaussian_nll(pred, logvar, target, min_var=1e-6):
    """Plain NLL, for reporting only."""
    return beta_nll(pred, logvar, target, beta=0.0, min_var=min_var)


# No target-marker pair is accepted as rigid without an anatomical validation.
RIGID_LINKS = ()


def link_variance(pred, links=RIGID_LINKS):
    """Temporal relative-length variance for explicitly validated marker links."""
    if not links:
        return pred.new_zeros(())
    a, b = zip(*links)
    d = (pred[..., list(a), :] - pred[..., list(b), :]).norm(dim=-1)   # (B,T,L)
    return (d.std(dim=1) / d.mean(dim=1).clamp_min(1e-6)).mean()
