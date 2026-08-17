"""Frozen fixed-lag construction of smoothed coordinates and their derivatives.

Follows the fixed-lag buffer pattern of Stanev et al. (10.3390/s21051804):
filter a trailing buffer, fit a cubic GCV smoothing spline, and evaluate value and
derivatives at a lagged physical time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import make_smoothing_spline

from posesim.shank_imu.signal import FixedLagSignal

_CONTRACT_PATH = Path(__file__).with_name("state_contract.json")


def hamming_sinc_lowpass(*, taps: int, cutoff_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Construct a unit-DC-gain odd-length Hamming-windowed-sinc low-pass FIR."""
    if not isinstance(taps, int) or taps < 3 or not taps % 2:
        raise ValueError("taps must be an odd integer of at least three")
    if not np.isfinite([cutoff_hz, sample_rate_hz]).all():
        raise ValueError("filter parameters must be finite")
    if not 0.0 < cutoff_hz < sample_rate_hz / 2.0:
        raise ValueError("cutoff_hz must lie inside the open Nyquist interval")
    samples = np.arange(taps, dtype=float) - (taps - 1) / 2.0
    coefficients = (
        2.0 * cutoff_hz / sample_rate_hz
        * np.sinc(2.0 * cutoff_hz / sample_rate_hz * samples)
        * np.hamming(taps)
    )
    return coefficients / coefficients.sum()


@dataclass(frozen=True)
class StateContract:
    sample_rate_hz: float
    buffer_samples: int
    evaluation_lag_samples: int
    group_delay_s: float
    fir_coefficients: np.ndarray


def load_state_contract(path: str | Path = _CONTRACT_PATH) -> StateContract:
    """Load the frozen constructor constants and verify the FIR coefficient hash."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != "shank-imu-state-contract-v1":
        raise ValueError("state contract schema is not shank-imu-state-contract-v1")
    fir = record["smoothing_fir"]
    coefficients = hamming_sinc_lowpass(
        taps=int(fir["taps"]),
        cutoff_hz=float(fir["cutoff_hz"]),
        sample_rate_hz=float(record["sample_rate_hz"]),
    )
    digest = hashlib.sha256(coefficients.astype(np.float64).tobytes()).hexdigest()
    if digest != fir["coefficients_sha256"]:
        raise ValueError("frozen FIR coefficients do not match their recorded hash")
    contract = StateContract(
        sample_rate_hz=float(record["sample_rate_hz"]),
        buffer_samples=int(record["buffer_samples"]),
        evaluation_lag_samples=int(record["evaluation_lag_samples"]),
        group_delay_s=float(record["group_delay_s"]),
        fir_coefficients=coefficients,
    )
    lag_s = contract.evaluation_lag_samples / contract.sample_rate_hz
    if not np.isclose(contract.group_delay_s, lag_s, rtol=0.0, atol=1e-12):
        raise ValueError("group_delay_s must equal the evaluation lag at the sample rate")
    if contract.buffer_samples < len(coefficients) + contract.evaluation_lag_samples:
        raise ValueError("buffer must cover the filter transient and the evaluation lag")
    return contract


def udot_noise_amplification(
    values: np.ndarray,
    time_s: np.ndarray,
    *,
    noise_std: float,
    seed: int,
    rotational: np.ndarray | None = None,
    contract: StateContract | None = None,
) -> dict:
    """Measure how one broadband-noise injection is amplified into `udot`.

    The GCV spline assumes independent residuals while the upstream FIR
    correlates them, so this gain is measured rather than derived; the formal
    acceptance bound is an outer-development choice made elsewhere.
    """
    if not np.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std must be one positive standard deviation")
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    noise = np.random.default_rng(seed).normal(scale=noise_std, size=values.shape)
    clean = fixed_lag_state(values, time_s, rotational=rotational, contract=contract)
    noisy = fixed_lag_state(values + noise, time_s, rotational=rotational, contract=contract)
    valid = clean.q.valid & noisy.q.valid
    if valid.sum() < 2:
        raise ValueError("need at least two valid frames to measure a variance")
    record: dict = {
        "noise_std": float(noise_std),
        "seed": int(seed),
        "frames_used": int(valid.sum()),
    }
    for name in ("q", "u", "udot"):
        delta = getattr(noisy, name).values[valid] - getattr(clean, name).values[valid]
        variance = delta.var(axis=0)
        record[f"{name}_delta_variance"] = [float(v) for v in variance]
        record[f"{name}_variance_amplification"] = [
            float(v / noise_std**2) for v in variance
        ]
    return record


@dataclass(frozen=True)
class FixedLagState:
    """Smoothed coordinates and their first two derivatives on one physical clock."""

    q: FixedLagSignal
    u: FixedLagSignal
    udot: FixedLagSignal


def fixed_lag_state(
    values: np.ndarray,
    time_s: np.ndarray,
    *,
    rotational: np.ndarray | None = None,
    contract: StateContract | None = None,
) -> FixedLagState:
    """Construct q, u, and udot from past samples only, at a fixed declared lag."""
    if contract is None:
        contract = load_state_contract()
    values = np.asarray(values, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or time.ndim != 1 or len(values) != len(time):
        raise ValueError("values and time_s must share one sample axis")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    if len(time) < contract.buffer_samples:
        raise ValueError("need at least one full buffer of samples")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing")
    step = np.diff(time)
    if not np.allclose(step, 1.0 / contract.sample_rate_hz, rtol=1e-6, atol=1e-9):
        raise ValueError("time_s must be uniform at the contract sample rate")
    if rotational is None:
        rotational = np.zeros(values.shape[1], dtype=bool)
    rotational = np.asarray(rotational, dtype=bool)
    if rotational.shape != (values.shape[1],):
        raise ValueError("rotational must give one flag per coordinate column")

    unwrapped = values.copy()
    unwrapped[:, rotational] = np.unwrap(values[:, rotational], axis=0)

    taps = len(contract.fir_coefficients)
    lag = contract.evaluation_lag_samples
    buffer = contract.buffer_samples
    filtered_span = buffer - taps + 1
    centre_offset = (taps - 1) // 2

    count, columns = unwrapped.shape
    streams = {
        name: np.zeros((count, columns), dtype=float) for name in ("q", "u", "udot")
    }
    valid = np.zeros(count, dtype=bool)
    for last in range(buffer - 1, count):
        physical = last - lag
        window = unwrapped[last - buffer + 1:last + 1]
        filtered = np.stack(
            [
                np.convolve(window[:, column], contract.fir_coefficients, mode="valid")
                for column in range(columns)
            ],
            axis=1,
        )
        filtered_time = time[
            last - buffer + 1 + centre_offset:
            last - buffer + 1 + centre_offset + filtered_span
        ]
        for column in range(columns):
            spline = make_smoothing_spline(filtered_time, filtered[:, column])
            streams["q"][physical, column] = spline(time[physical])
            streams["u"][physical, column] = spline.derivative(1)(time[physical])
            streams["udot"][physical, column] = spline.derivative(2)(time[physical])
        valid[physical] = True

    available = time + contract.group_delay_s
    signals = {
        name: FixedLagSignal(stream, time.copy(), available.copy(), valid.copy())
        for name, stream in streams.items()
    }
    return FixedLagState(q=signals["q"], u=signals["u"], udot=signals["udot"])
