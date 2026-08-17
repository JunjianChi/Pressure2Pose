"""Validate a MovePort aligned cache or build one raw segment in memory for QC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from posesim.data.moveport import load_native_segment
from posesim.data.mpdataset import (
    align_native_segment,
    aligned_cache_payload,
    project_source_mask,
    subject_cell_area_native,
    validate_aligned_cache,
)


def _load(args, parser):
    if args.cache:
        with np.load(args.cache, allow_pickle=False) as archive:
            validate_aligned_cache(archive)
            return {key: archive[key].copy() for key in archive.files}
    missing = [name for name in ("subject", "activity", "segment") if getattr(args, name) is None]
    if missing:
        parser.error("--root requires --subject, --activity, and --segment")
    native = load_native_segment(args.root, args.subject, args.activity, args.segment)
    source_mask = project_source_mask(args.root)
    area = subject_cell_area_native(args.root, args.subject)
    if area is None:
        parser.error("no native equal-rate pressure/force pair can establish cell area")
    aligned = align_native_segment(native, args.root, source_mask, area)
    return aligned_cache_payload([aligned], n_folds=1, seed=0)


def _summary(cache):
    validity = {}
    for name in ("pressure", "force", "foot_imu", "target", "contact"):
        validity[name] = float(np.asarray(cache[f"{name}_valid"]).mean())
    return {
        "schema_version": np.asarray(cache["schema_version"]).item(),
        "alignment_status": np.asarray(cache["alignment_status"]).item(),
        "segments": int(len(cache["segment_id"])),
        "unique_frames": int(len(cache["time_s"])),
        "common_group_delay_s": float(cache["common_group_delay_s"]),
        "valid_fraction": validity,
    }


def _reserve(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", help="existing moveport-aligned-v2 NPZ")
    source.add_argument("--root", help="MovePort release root for a one-segment in-memory smoke")
    parser.add_argument("--subject")
    parser.add_argument("--activity")
    parser.add_argument("--segment")
    parser.add_argument("--report", help="optional JSON report path; no report is written by default")
    parser.add_argument("--figure", help="optional validity plot path; no figure is written by default")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache = _load(args, parser)
    summary = _summary(cache)
    if args.report:
        destination = _reserve(args.report, args.overwrite)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
    if args.figure:
        import matplotlib.pyplot as plt
        destination = _reserve(args.figure, args.overwrite)
        names = list(summary["valid_fraction"])
        values = [summary["valid_fraction"][name] for name in names]
        figure, axis = plt.subplots(figsize=(6, 3))
        axis.bar(names, values)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("valid fraction")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
