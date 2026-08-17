"""Plot validation curves from the history files train.py writes.

    python scripts/plot_training.py checkpoints/unet_lab.pt.history.json \\
        checkpoints/mlp_lab.pt.history.json --out training_curves.png
"""

from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("histories", nargs="+")
    ap.add_argument("--out", default="training_curves.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    fig, (ax_mse, ax_r2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for path in args.histories:
        with open(path) as fh:
            record = json.load(fh)
        epochs = [e["epoch"] for e in record["epochs"]]
        ax_mse.plot(epochs, [e["val_mse"] for e in record["epochs"]], label=record["model"], linewidth=2)
        ax_r2.plot(epochs, [e["val_r2"] for e in record["epochs"]], label=record["model"], linewidth=2)

    ax_mse.set_xlabel("epoch")
    ax_mse.set_ylabel("validation masked MSE")
    ax_mse.set_yscale("log")
    ax_r2.set_xlabel("epoch")
    ax_r2.set_ylabel("validation R²")
    for ax in (ax_mse, ax_r2):
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
