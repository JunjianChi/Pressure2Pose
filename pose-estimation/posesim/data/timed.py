"""Immutable sampled arrays with explicit timing and validity metadata."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_scalar(value, name):
    scalar = np.asarray(value)
    if scalar.ndim != 0 or not np.issubdtype(scalar.dtype, np.number) or np.issubdtype(scalar.dtype, np.bool_):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(scalar)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


@dataclass(frozen=True)
class TimedArray:
    values: np.ndarray
    time_s: np.ndarray
    valid: np.ndarray
    unit: str
    time_basis: str
    nominal_hz: float
    group_delay_s: float = 0.0

    def __post_init__(self):
        values = np.array(self.values, copy=True)
        time_s = np.array(self.time_s, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=bool, copy=True)
        if values.ndim < 1:
            raise ValueError("values must have a sample axis")
        if time_s.ndim != 1 or time_s.shape[0] != values.shape[0]:
            raise ValueError("time_s must match the values sample axis")
        if valid.shape != values.shape:
            raise ValueError("valid must match values")
        if not np.isfinite(time_s).all() or (time_s.size > 1 and np.any(np.diff(time_s) <= 0)):
            raise ValueError("time_s must be finite and strictly increasing")
        if np.any((~np.isfinite(values)) & valid):
            raise ValueError("nonfinite values must be invalid")
        if type(self.unit) is not str or not self.unit:
            raise ValueError("unit must be a non-empty Python string")
        if type(self.time_basis) is not str or not self.time_basis:
            raise ValueError("time_basis must be a non-empty Python string")
        nominal_hz = _finite_scalar(self.nominal_hz, "nominal_hz")
        group_delay_s = _finite_scalar(self.group_delay_s, "group_delay_s")
        if nominal_hz <= 0:
            raise ValueError("nominal_hz must be positive")
        values.setflags(write=False)
        time_s.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "nominal_hz", nominal_hz)
        object.__setattr__(self, "group_delay_s", group_delay_s)
