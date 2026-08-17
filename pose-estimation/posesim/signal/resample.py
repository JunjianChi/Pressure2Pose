"""Causal anti-aliased resampling for explicitly timed arrays."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from posesim.data.timed import TimedArray

GROUP_DELAY_S = 0.1
HISTORY_S = 2.0 * GROUP_DELAY_S


def _uniform_rate(time_s, declared_hz, label):
    if len(time_s) < 2:
        return float(declared_hz)
    steps = np.diff(time_s)
    expected = 1.0 / float(declared_hz)
    if not np.allclose(steps, expected, rtol=1e-6, atol=1e-9):
        raise ValueError(f"{label} time must be uniform at its declared nominal rate")
    return float(declared_hz)


def _target_rate(time_s, fallback):
    if len(time_s) < 2:
        return float(fallback)
    steps = np.diff(time_s)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        raise ValueError("target time must be uniform")
    return float(1.0 / steps[0])


def _digital_gain(offset, sigma, frequency_hz):
    kernel = np.exp(-0.5 * (offset / sigma) ** 2)
    kernel /= kernel.sum()
    return abs(np.sum(kernel * np.exp(1j * 2.0 * np.pi * frequency_hz * offset)))


@lru_cache(maxsize=32)
def _rate_calibrated_sigma(sample_hz, cutoff_hz):
    count = int(round(HISTORY_S * sample_hz))
    if not np.isclose(count / sample_hz, HISTORY_S, rtol=0.0, atol=1e-12):
        raise ValueError("source rate must represent the fixed filter history exactly")
    offset = (np.arange(count + 1) - count / 2.0) / sample_hz
    low, high = 1e-8, GROUP_DELAY_S
    if _digital_gain(offset, high, cutoff_hz) > 0.5:
        raise ValueError("filter history is too short for half-amplitude cutoff calibration")
    for _ in range(80):
        middle = (low + high) / 2.0
        if _digital_gain(offset, middle, cutoff_hz) > 0.5:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _weights(sample_time_s, output_time_s, cutoff_hz, source_hz):
    centre = output_time_s - GROUP_DELAY_S
    offset = sample_time_s - centre
    sigma = _rate_calibrated_sigma(float(source_hz), float(cutoff_hz))
    kernel = np.exp(-0.5 * (offset / sigma) ** 2)
    total = kernel.sum()
    if abs(total) < 1e-12:
        raise ValueError("causal resampling kernel has zero gain")
    return kernel / total


def resample_causal(signal: TimedArray, target_time_s, cutoff_hz) -> TimedArray:
    """Low-pass and sample a uniform stream using a fixed physical delay.

    Every output at time ``t`` uses source samples only in ``[t-0.2, t]``.
    The kernel is centred at ``t-0.1`` and its width is calibrated against the
    native-rate digital response. Direct evaluation at ``t`` gives 60 and 100
    Hz sources the same half-amplitude cutoff and 0.1-second delay without a
    hidden zero-order-hold phase. Missing history or any invalid contributing
    value makes that output invalid and NaN.
    """
    target = np.asarray(target_time_s, dtype=np.float64)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError("target_time_s must be a finite one-dimensional array")
    if target.size > 1 and np.any(np.diff(target) <= 0.0):
        raise ValueError("target_time_s must be strictly increasing")
    source_hz = _uniform_rate(signal.time_s, signal.nominal_hz, "source")
    target_hz = _target_rate(target, source_hz)
    cutoff_hz = float(cutoff_hz)
    if (not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0
            or cutoff_hz >= min(source_hz, target_hz) / 2.0):
        raise ValueError("cutoff_hz must lie below both source and target Nyquist frequencies")

    out_shape = (len(target),) + signal.values.shape[1:]
    out = np.full(out_shape, np.nan, dtype=np.float64)
    valid = np.zeros(out_shape, dtype=bool)
    if not len(signal.time_s):
        return TimedArray(out, target, valid, signal.unit,
                          f"{signal.time_basis};causal_fir_exact_target_time", target_hz,
                          signal.group_delay_s + GROUP_DELAY_S)

    tolerance = 1e-9
    for output_index, time_s in enumerate(target):
        history_start = time_s - HISTORY_S
        if history_start < signal.time_s[0] - tolerance or time_s > signal.time_s[-1] + tolerance:
            continue
        first = np.searchsorted(signal.time_s, history_start - tolerance, side="left")
        stop = np.searchsorted(signal.time_s, time_s + tolerance, side="right")
        sample_time = signal.time_s[first:stop]
        sample_valid = signal.valid[first:stop]
        channel_valid = np.all(sample_valid, axis=0)
        weights = _weights(sample_time, time_s, cutoff_hz, source_hz)
        values = np.tensordot(weights, np.where(sample_valid, signal.values[first:stop], 0.0), axes=(0, 0))
        out[output_index] = values
        valid[output_index] = channel_valid
    out[~valid] = np.nan
    return TimedArray(out, target, valid, signal.unit,
                      f"{signal.time_basis};causal_fir_exact_target_time", target_hz,
                      signal.group_delay_s + GROUP_DELAY_S)
