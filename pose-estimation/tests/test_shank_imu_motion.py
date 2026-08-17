"""Admission tests for the OpenSim motion-storage coordinate reader."""
from __future__ import annotations

import numpy as np
import pytest

from posesim.shank_imu.motion import read_motion

HEADER = """Coordinates
version=1
nRows={rows}
nColumns={columns}
inDegrees={degrees}
endheader
time\tpelvis_tilt\tpelvis_tx
"""


def _write(tmp_path, rows, *, degrees="yes"):
    body = "\n".join("\t".join(f"{value:.8f}" for value in row) for row in rows)
    path = tmp_path / "ik.mot"
    path.write_text(
        HEADER.format(rows=len(rows), columns=3, degrees=degrees) + body + "\n",
        encoding="utf-8",
    )
    return path


def test_reads_time_labels_values_and_degree_flag(tmp_path):
    rows = [[0.00, 12.5, 0.301], [0.01, 13.0, 0.302], [0.02, 13.5, 0.303]]
    time, values, labels, in_degrees = read_motion(_write(tmp_path, rows))
    assert labels == ("pelvis_tilt", "pelvis_tx")
    assert in_degrees is True
    assert np.allclose(time, [0.00, 0.01, 0.02])
    assert np.allclose(values, np.asarray(rows)[:, 1:])


def test_radian_files_report_their_flag(tmp_path):
    rows = [[0.00, 0.2, 0.3], [0.01, 0.2, 0.3]]
    _, _, _, in_degrees = read_motion(_write(tmp_path, rows, degrees="no"))
    assert in_degrees is False


def test_rejects_malformed_storage(tmp_path):
    path = tmp_path / "broken.mot"
    path.write_text("Coordinates\nnRows=1\nnColumns=3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_motion(path)
    rows = [[0.00, 1.0, 2.0], [0.01, 1.0]]
    body = "\n".join("\t".join(f"{v:.4f}" for v in row) for row in rows)
    ragged = tmp_path / "ragged.mot"
    ragged.write_text(HEADER.format(rows=2, columns=3, degrees="yes") + body + "\n",
                      encoding="utf-8")
    with pytest.raises(ValueError):
        read_motion(ragged)
    missing_flag = tmp_path / "noflag.mot"
    missing_flag.write_text(
        "Coordinates\nnRows=1\nnColumns=3\nendheader\ntime\ta\tb\n0.0\t1.0\t2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_motion(missing_flag)
    nonfinite = _write(tmp_path, [[0.00, np.nan, 0.3], [0.01, 0.2, 0.3]])
    with pytest.raises(ValueError):
        read_motion(nonfinite)
