"""Convert the stacked crossbar/diode capture into pipeline pair archives.

The source directory holds `input_{train,test}.npy` (crossbar readouts, the
crosstalk-contaminated input) paired frame-for-frame with `label_{train,test}.npy`
(diode-array references), on the 33x15 array in ohms. That split was made upstream
by a seeded frame-level `random_split`, so the capture order is not recoverable and
the test set is not temporally independent of training.

The capture itself is not released. `scripts/make_synthetic.py` builds archives in
the same format from the simulator, which is what the pipeline runs on by default.

    python scripts/import_lab_data.py --source /path/to/paired_archives
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from crosstalk.data import pressed_from_grid


def convert(source: str, out_dir: str) -> None:
    for split in ("train", "test"):
        ct_r = np.load(os.path.join(source, f"input_{split}.npy"))
        ref_r = np.load(os.path.join(source, f"label_{split}.npy"))
        if ct_r.shape != ref_r.shape:
            raise ValueError(f"{split}: input and label shapes differ")
        out = os.path.join(out_dir, f"lab_{split}.npz")
        np.savez(out, ct=pressed_from_grid(ct_r), ref=pressed_from_grid(ref_r))
        print(f"{split}: {len(ct_r)} pairs -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="directory with the paired .npy arrays")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    convert(args.source, args.out_dir)


if __name__ == "__main__":
    main()
