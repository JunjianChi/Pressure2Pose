"""Build simulated train and test archives, so training runs without the lab capture.

    python scripts/make_synthetic.py --n-train 1600 --n-test 400
"""

from __future__ import annotations

import argparse
import os
import numpy as np

from crosstalk.data import pressed_from_grid
from crosstalk.synthetic import make_pairs


def _progress(label: str, every: int = 100):
    def report(done: int, total: int):
        if done % every == 0 or done == total:
            print(f"{label}: {done}/{total} frames", flush=True)
    return report


def build(split: str, n: int, seed: int, out_dir: str) -> str:
    measured, truth = make_pairs(n, seed=seed, progress=_progress(split))
    out = os.path.join(out_dir, f"synthetic_{split}.npz")
    np.savez(out, ct=pressed_from_grid(measured), ref=pressed_from_grid(truth))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split, n, seed in (("train", args.n_train, args.seed), ("test", args.n_test, args.seed + 1)):
        if n > 0:
            print(f"wrote {n} pairs to {build(split, n, seed, args.out_dir)}")


if __name__ == "__main__":
    main()
