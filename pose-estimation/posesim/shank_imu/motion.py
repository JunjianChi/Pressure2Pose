"""Reader for OpenSim motion-storage coordinate trajectories."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_motion(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], bool]:
    """Read one `.mot` coordinate file as (time, values, labels, in_degrees).

    The angular unit flag is returned, never applied: converting rotational
    columns needs the model's coordinate motion types, which the caller owns.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        header_end = lines.index("endheader")
    except ValueError as error:
        raise ValueError("OpenSim motion storage has no endheader") from error
    in_degrees = None
    for line in lines[:header_end]:
        key, _, value = line.partition("=")
        if key.strip().lower() == "indegrees":
            in_degrees = value.strip().lower()
    if in_degrees not in ("yes", "no"):
        raise ValueError("OpenSim motion storage must declare inDegrees=yes or no")
    if header_end + 2 > len(lines):
        raise ValueError("OpenSim motion storage has no column header or data")
    columns = lines[header_end + 1].split()
    if len(columns) < 2 or columns[0] != "time":
        raise ValueError("OpenSim motion storage must lead with a time column")
    rows = []
    for line in lines[header_end + 2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(columns):
            raise ValueError("OpenSim motion storage row does not match its header")
        rows.append([float(field) for field in fields])
    if not rows:
        raise ValueError("OpenSim motion storage has no data rows")
    table = np.asarray(rows, dtype=float)
    if not np.isfinite(table).all():
        raise ValueError("OpenSim motion storage values must be finite")
    return (
        table[:, 0],
        table[:, 1:],
        tuple(columns[1:]),
        in_degrees == "yes",
    )
