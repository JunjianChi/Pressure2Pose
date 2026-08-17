"""Per-segment IK marker residuals, keyed as the streaming report keys them.

Reads the `_ik_marker_errors.sto` files the batch run copied beside each cache
and reduces each to one within-trial mean RMS in metres, the quantity
`segment_fold_cutoffs` turns into a per-fold acceptance cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PREFIX = "ik_errors_"
RMS_COLUMN = "marker_error_RMS"


def read_marker_errors(path: Path) -> np.ndarray:
    """The RMS column of one OpenSim marker-error storage file, in metres."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "endheader":
            break
    else:
        raise ValueError(f"{path} has no endheader line")
    columns = lines[index + 1].split()
    if RMS_COLUMN not in columns:
        raise ValueError(f"{path} has no {RMS_COLUMN} column")
    which = columns.index(RMS_COLUMN)
    values = np.asarray([float(row.split()[which]) for row in lines[index + 2:] if row.strip()])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{path} has no finite residual rows")
    return values


def cache_segment_keys(cache) -> dict[str, str]:
    """Filename stem to `subject/activity/name`, the key the streaming report uses.

    Both fields carry underscores of their own (`treadmill_normal` / `high_1`),
    so the split is read off the cache's segment table rather than guessed.
    """
    keys: dict[str, str] = {}
    for subject, activity, name in zip(cache["segment_subject"], cache["segment_activity"],
                                       cache["segment_name"]):
        stem = f"{subject}_{activity}_{name}"
        if stem in keys:
            raise ValueError(f"two cache segments share the stem {stem}")
        keys[stem] = f"{subject}/{activity}/{name}"
    if not keys:
        raise ValueError("the cache lists no segments")
    return keys


def collect(directory: Path, segment_keys: dict[str, str]) -> dict[str, dict]:
    paths = sorted(directory.glob(f"{PREFIX}*.sto"))
    if not paths:
        raise ValueError(f"no marker-error files under {directory}")
    rows: dict[str, dict] = {}
    for path in paths:
        stem = path.stem[len(PREFIX):] if path.stem.startswith(PREFIX) else None
        key = segment_keys.get(stem) if stem else None
        if key is None:
            raise ValueError(f"{path} does not name a segment of the aligned cache")
        if key in rows:
            raise ValueError(f"two marker-error files map to {key}")
        values = read_marker_errors(path)
        rows[key] = {
            "subject": key.split("/")[0],
            "frames": int(len(values)),
            "mean_rms_m": float(values.mean()),
            "p95_rms_m": float(np.percentile(values, 95.0)),
            "max_rms_m": float(values.max()),
        }
    missing = sorted(set(segment_keys.values()) - set(rows))
    if missing:
        raise ValueError(f"no marker-error file for {len(missing)} cache segments: {missing[:5]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True,
                        help="aligned cache whose segment table names the keys")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=True) as cache:
        segment_keys = cache_segment_keys(cache)
    rows = collect(args.errors_dir, segment_keys)
    means = np.asarray([row["mean_rms_m"] for row in rows.values()])
    payload = {
        "schema": "segment-fit-quality-v1",
        "segments": len(rows),
        "cohort_mean_rms_m": {
            "median": float(np.median(means)),
            "p95": float(np.percentile(means, 95.0)),
            "max": float(means.max()),
        },
        "per_segment": rows,
    }
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"segments": payload["segments"],
                      **payload["cohort_mean_rms_m"]}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
