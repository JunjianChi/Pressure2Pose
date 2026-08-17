"""Readers for declared scalar metrics in OpenSim acceptance reports."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def marker_pair_summary(
    markers: np.ndarray, labels: list[str], first: str, second: str
) -> dict[str, float | int]:
    """Summarise one marker-pair length without treating it as a rigid-link proof."""
    values = np.asarray(markers, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise ValueError("markers must be a finite array with shape (frames, markers, 3)")
    if values.shape[1] != len(labels) or len(labels) != len(set(labels)):
        raise ValueError("labels must be unique and match the marker axis")
    try:
        first_index, second_index = labels.index(first), labels.index(second)
    except ValueError as error:
        raise ValueError("marker-pair labels must resolve") from error
    lengths = np.linalg.norm(values[:, first_index] - values[:, second_index], axis=1)
    median = float(np.median(lengths))
    if median <= 0.0:
        raise ValueError("marker-pair median length must be positive")
    return {
        "frames": int(len(lengths)),
        "median_length_m": median,
        "max_length_m": float(lengths.max()),
        "max_relative_deviation": float(np.abs(lengths - median).max() / median),
    }


def marker_pair_reports(
    markers: np.ndarray, labels: list[str], pairs: list[tuple[str, str, str]]
) -> list[dict[str, float | int | str]]:
    """Report declared marker-pair summaries in the supplied order."""
    names = [name for name, _, _ in pairs]
    if not names or len(names) != len(set(names)):
        raise ValueError("marker-pair names must be non-empty and unique")
    reports: list[dict[str, float | int | str]] = []
    for name, first, second in pairs:
        report: dict[str, float | int | str] = {"name": name}
        report.update(marker_pair_summary(markers, labels, first, second))
        reports.append(report)
    return reports


def marker_pair_validity(
    markers: np.ndarray,
    labels: list[str],
    pairs: list[tuple[str, str, str]],
    *,
    max_relative_deviation: float,
) -> np.ndarray:
    """Mark frames that stay within one supplied source-marker excursion limit."""
    values = np.asarray(markers, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise ValueError("markers must be a finite array with shape (frames, markers, 3)")
    if values.shape[1] != len(labels) or len(labels) != len(set(labels)):
        raise ValueError("labels must be unique and match the marker axis")
    if not np.isfinite(max_relative_deviation) or max_relative_deviation < 0.0:
        raise ValueError("max_relative_deviation must be finite and non-negative")
    valid = np.ones(len(values), dtype=bool)
    for _, first, second in pairs:
        try:
            first_index, second_index = labels.index(first), labels.index(second)
        except ValueError as error:
            raise ValueError("marker-pair labels must resolve") from error
        lengths = np.linalg.norm(values[:, first_index] - values[:, second_index], axis=1)
        median = float(np.median(lengths))
        if median <= 0.0:
            raise ValueError("marker-pair median length must be positive")
        valid &= np.abs(lengths - median) / median <= max_relative_deviation
    return valid


def above_band_energy_share(
    signals: np.ndarray, *, sample_rate_hz: float, cutoff_hz: float
) -> np.ndarray:
    """Share of mean-removed spectral energy above a cutoff, per channel.

    A provenance diagnostic for archived channels: it bounds what a
    construction filter discards without entering any model input.
    """
    values = np.asarray(signals, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 8 or not np.isfinite(values).all():
        raise ValueError("signals must be one finite (frames, channels) array")
    if not np.isfinite([sample_rate_hz, cutoff_hz]).all() or not (
            0.0 < cutoff_hz < sample_rate_hz / 2.0):
        raise ValueError("cutoff_hz must lie inside the open Nyquist interval")
    centred = values - values.mean(axis=0)
    spectrum = np.abs(np.fft.rfft(centred, axis=0)) ** 2
    frequency_hz = np.fft.rfftfreq(len(centred), d=1.0 / sample_rate_hz)
    total = spectrum.sum(axis=0)
    if np.any(total <= 0.0):
        raise ValueError("every channel needs non-zero energy after mean removal")
    return spectrum[frequency_hz > cutoff_hz].sum(axis=0) / total


def ik_error_summary(path: str | Path) -> dict[str, float | int]:
    """Summarise the OpenSim IK marker-RMS column in metres."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        header_end = lines.index("endheader")
    except ValueError as error:
        raise ValueError("OpenSim storage report has no endheader") from error
    if header_end + 2 > len(lines):
        raise ValueError("OpenSim storage report has no column header or data")
    columns = lines[header_end + 1].split()
    try:
        rms_index = columns.index("marker_error_RMS")
        max_index = columns.index("marker_error_max")
    except ValueError as error:
        raise ValueError("OpenSim storage report lacks a declared marker-error column") from error
    rms_values = []
    max_values = []
    for line in lines[header_end + 2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(columns):
            raise ValueError("OpenSim storage report row does not match its header")
        rms_values.append(float(fields[rms_index]))
        max_values.append(float(fields[max_index]))
    rms = np.asarray(rms_values, dtype=float)
    maximum = np.asarray(max_values, dtype=float)
    if (
        not len(rms)
        or not np.isfinite(rms).all()
        or not np.isfinite(maximum).all()
        or np.any(rms < 0.0)
        or np.any(maximum < 0.0)
    ):
        raise ValueError("marker_error_RMS must contain finite non-negative values")
    return {
        "frames": int(len(rms)),
        "mean_rms_m": float(rms.mean()),
        "p95_rms_m": float(np.quantile(rms, 0.95)),
        "max_marker_error_m": float(maximum.max()),
    }
