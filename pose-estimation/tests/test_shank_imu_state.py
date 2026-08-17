"""Admission tests for the frozen fixed-lag q -> u -> udot state constructor."""
from __future__ import annotations

import numpy as np
import pytest

from posesim.shank_imu.state import (
    fixed_lag_state,
    hamming_sinc_lowpass,
    load_state_contract,
)

FS = 100.0


def _time(n: int) -> np.ndarray:
    return np.arange(n) / FS


def test_contract_loads_and_verifies_frozen_coefficients():
    contract = load_state_contract()
    assert contract.sample_rate_hz == FS
    assert contract.buffer_samples == 35
    assert contract.evaluation_lag_samples == 14
    assert contract.group_delay_s == pytest.approx(0.14)
    coefficients = contract.fir_coefficients
    assert len(coefficients) == 15
    assert coefficients.sum() == pytest.approx(1.0)
    assert np.allclose(coefficients, coefficients[::-1])


def test_fir_rejects_invalid_designs():
    with pytest.raises(ValueError):
        hamming_sinc_lowpass(taps=14, cutoff_hz=6.0, sample_rate_hz=FS)
    with pytest.raises(ValueError):
        hamming_sinc_lowpass(taps=15, cutoff_hz=0.0, sample_rate_hz=FS)
    with pytest.raises(ValueError):
        hamming_sinc_lowpass(taps=15, cutoff_hz=60.0, sample_rate_hz=FS)


def test_constant_pose_is_static():
    n = 120
    q = np.full((n, 2), (0.7, -1.2)).astype(float)
    state = fixed_lag_state(q, _time(n))
    valid = state.q.valid
    assert valid.any()
    assert np.allclose(state.q.values[valid], q[valid], atol=1e-9)
    assert np.allclose(state.u.values[valid], 0.0, atol=1e-7)
    assert np.allclose(state.udot.values[valid], 0.0, atol=1e-5)


def test_ramp_recovers_velocity():
    n = 150
    time = _time(n)
    q = np.outer(time, [0.9, -0.4])
    state = fixed_lag_state(q, time)
    valid = state.u.valid
    assert np.allclose(state.u.values[valid], [0.9, -0.4], atol=1e-6)
    assert np.allclose(state.udot.values[valid], 0.0, atol=1e-4)


def test_sinusoid_derivatives_match_analytic():
    n = 300
    time = _time(n)
    omega = 2.0 * np.pi * 1.0
    q = np.sin(omega * time)[:, None]
    state = fixed_lag_state(q, time)
    valid = state.q.valid
    assert np.allclose(state.u.values[valid, 0], omega * np.cos(omega * time[valid]),
                       atol=0.02 * omega)
    assert np.allclose(state.udot.values[valid, 0], -omega**2 * np.sin(omega * time[valid]),
                       atol=0.05 * omega**2)


def test_prefix_invariance_at_available_time():
    rng = np.random.default_rng(7)
    n = 140
    time = _time(n)
    q = np.cumsum(rng.normal(scale=0.01, size=(n, 1)), axis=0)
    full = fixed_lag_state(q, time)
    cut = 100
    truncated = fixed_lag_state(q[:cut], time[:cut])
    usable = truncated.q.valid
    assert usable.any()
    for stream in ("q", "u", "udot"):
        full_signal = getattr(full, stream)
        cut_signal = getattr(truncated, stream)
        assert np.array_equal(cut_signal.values[usable], full_signal.values[:cut][usable])


def test_timing_and_validity_edges():
    n = 90
    time = _time(n)
    state = fixed_lag_state(np.zeros((n, 1)), time)
    for signal in (state.q, state.u, state.udot):
        assert np.array_equal(signal.physical_time_s, time)
        assert np.allclose(signal.available_time_s, time + 0.14)
        expected = np.zeros(n, dtype=bool)
        expected[20:n - 14] = True
        assert np.array_equal(signal.valid, expected)


def test_rotational_unwrap_gives_continuous_rate():
    n = 200
    time = _time(n)
    rate = 5.0
    wrapped = np.angle(np.exp(1j * rate * time))[:, None]
    rotational = fixed_lag_state(wrapped, time, rotational=np.array([True]))
    valid = rotational.u.valid
    assert np.allclose(rotational.u.values[valid, 0], rate, atol=1e-3)
    raw = fixed_lag_state(wrapped, time, rotational=np.array([False]))
    assert np.abs(raw.u.values[valid, 0]).max() > 10.0 * rate


def test_noise_amplification_measures_the_udot_variance_gain():
    from posesim.shank_imu.state import udot_noise_amplification

    n = 400
    time = _time(n)
    clean = 0.3 * np.sin(2.0 * np.pi * 1.0 * time)[:, None]
    small = udot_noise_amplification(clean, time, noise_std=1e-3, seed=0)
    assert small["noise_std"] == pytest.approx(1e-3)
    assert small["frames_used"] > 200
    gain = small["udot_variance_amplification"][0]
    assert np.isfinite(gain) and gain > 1.0
    assert small["q_variance_amplification"][0] < 1.0

    repeat = udot_noise_amplification(clean, time, noise_std=1e-3, seed=0)
    assert repeat["udot_variance_amplification"][0] == pytest.approx(gain)

    larger = udot_noise_amplification(clean, time, noise_std=2e-3, seed=0)
    ratio = larger["udot_delta_variance"][0] / small["udot_delta_variance"][0]
    assert 2.0 < ratio < 8.0


def test_input_validation():
    n = 60
    time = _time(n)
    with pytest.raises(ValueError):
        fixed_lag_state(np.zeros((n, 1)), time[::-1])
    with pytest.raises(ValueError):
        fixed_lag_state(np.zeros((n, 1)), np.concatenate([time[:30], time[31:]])[:n - 1])
    bad = np.zeros((n, 1))
    bad[5, 0] = np.nan
    with pytest.raises(ValueError):
        fixed_lag_state(bad, time)
    with pytest.raises(ValueError):
        fixed_lag_state(np.zeros((n, 1)), time, rotational=np.array([True, False]))
    with pytest.raises(ValueError):
        fixed_lag_state(np.zeros((34, 1)), _time(34))
