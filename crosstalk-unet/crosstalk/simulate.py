"""Physics simulator for crossbar crosstalk: nodal analysis of the resistor grid.

Each sensor is a resistor between its row and column electrode. Reading one
sensor drives its row through a series resistor and grounds its column while
every other electrode floats, so current also flows through neighbouring
sensors (sneak paths) and the measured resistance is contaminated. Simulating
that readout over a ground-truth resistance map produces paired
(crosstalk, reference) frames without a physical capture, though its
potential-divider excitation is not the Howland current source of the target
insole: what it reproduces is the sneak-path mechanism, not that front end.
"""

from __future__ import annotations

import numpy as np


class CrossbarReadout:
    """Potential-divider readout of a rows x cols resistive crossbar."""

    def __init__(self, r_series: float = 300.0, vcc: float = 0.2):
        if r_series <= 0.0 or vcc <= 0.0:
            raise ValueError("r_series and vcc must be positive")
        self.r_series = float(r_series)
        self.vcc = float(vcc)

    @staticmethod
    def _conductance(r_matrix: np.ndarray) -> np.ndarray:
        """Laplacian of the two-layer electrode graph: rows first, then columns."""
        rows, cols = r_matrix.shape
        g = 1.0 / r_matrix
        lap = np.zeros((rows + cols, rows + cols))
        lap[:rows, :rows] = np.diag(g.sum(axis=1))
        lap[rows:, rows:] = np.diag(g.sum(axis=0))
        lap[:rows, rows:] = -g
        lap[rows:, :rows] = -g.T
        return lap

    def measure_cell(self, r_matrix: np.ndarray, row: int, col: int) -> float:
        """Measured resistance of one cell, sneak paths included."""
        rows, cols = r_matrix.shape
        g = self._conductance(r_matrix)
        b = np.zeros(rows + cols)

        # Drive the selected row from vcc through the series resistor.
        g[row, row] += 1.0 / self.r_series
        b[row] = self.vcc / self.r_series

        # Ground the selected column electrode.
        ground = rows + col
        g[ground, :] = 0.0
        g[:, ground] = 0.0
        g[ground, ground] = 1.0
        b[ground] = 0.0

        v = np.linalg.solve(g, b)
        v_measured = v[row] - v[ground]
        if np.isclose(v_measured, self.vcc):
            return np.inf
        return v_measured * self.r_series / (self.vcc - v_measured)

    def measure(self, r_matrix: np.ndarray) -> np.ndarray:
        """Measured resistance of every cell of a ground-truth resistance map."""
        r_matrix = np.asarray(r_matrix, dtype=np.float64)
        if r_matrix.ndim != 2 or (r_matrix <= 0.0).any() or not np.isfinite(r_matrix).all():
            raise ValueError("r_matrix must be a 2-D array of positive finite resistances")
        out = np.empty_like(r_matrix)
        for i in range(r_matrix.shape[0]):
            for j in range(r_matrix.shape[1]):
                out[i, j] = self.measure_cell(r_matrix, i, j)
        return out
