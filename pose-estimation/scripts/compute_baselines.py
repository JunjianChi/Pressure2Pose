"""Score the mean-pose and nearest-pressure-retrieval baselines for every fold.

Both belong beside every result table, so they are scored once here and the
formal report carries them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from posesim.analysis.baselines import fold_baselines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/processed/moveport_all.npz"))
    parser.add_argument("--retrieval-stride", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as archive:
        cache = {key: archive[key].copy() for key in archive.files}
    folds = len(np.asarray(cache["fold_subjects"]))
    result = {}
    for fold in range(folds):
        result[str(fold)] = fold_baselines(cache, fold,
                                           retrieval_stride=args.retrieval_stride)
        print(json.dumps({
            "fold": fold,
            "mean_pose_mm": round(result[str(fold)]["mean_pose"]["subject_averaged_mpjpe_mm"], 1),
            "retrieval_mm": round(result[str(fold)]["retrieval"]["subject_averaged_mpjpe_mm"], 1),
        }, sort_keys=True), flush=True)
    cohort = {
        name: float(np.mean([result[str(f)][name]["subject_averaged_mpjpe_mm"]
                             for f in range(folds)]))
        for name in ("mean_pose", "retrieval")
    }
    args.out.write_text(json.dumps({"per_fold": result, "cohort_mean_mm": cohort},
                                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cohort_mean_mm": cohort}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
