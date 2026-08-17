"""Build the training .npz from a raw capture CSV.

    python scripts/prepare_data.py --csv data/raw/insole_frames_flat.csv --out data/processed/pairs.npz
"""

from __future__ import annotations

import argparse
import os

from crosstalk.data import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/insole_frames_flat.csv")
    ap.add_argument("--out", default="data/processed/pairs.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    info = prepare(args.csv, args.out)
    print(f"wrote {info['n_pairs']} pairs to {info['out']}")


if __name__ == "__main__":
    main()
