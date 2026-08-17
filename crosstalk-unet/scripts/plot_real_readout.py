"""Render one captured frame pair: crossbar readout beside its diode reference.

    python scripts/plot_real_readout.py --source /path/to/paired_archives --index 2037 --out readout.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from crosstalk.sensor import COLS, ROWS, active_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="directory with the paired .npy arrays")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--schematic", default="", help="optional circuit-schematic image for a first panel")
    ap.add_argument("--out", default="readout.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    crossbar = np.load(os.path.join(args.source, f"input_{args.split}.npy"))
    diode = np.load(os.path.join(args.source, f"label_{args.split}.npy")).astype(float)
    mask = active_mask()

    panels = [
        ("crossbar readout (ohm)", crossbar[args.index]),
        ("diode reference (ohm)", diode[args.index]),
    ]
    if args.schematic:
        fig, axes = plt.subplots(
            1, 3, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1]}
        )
        axes[0].imshow(plt.imread(args.schematic))
        axes[0].set_title("sneak paths", fontsize=10)
        axes[0].axis("off")
        axes = axes[1:]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.6))
    for ax, (title, grid) in zip(axes, panels):
        lo, hi = np.percentile(grid[mask], [1.0, 99.0])
        display = np.where(mask, grid, np.nan)
        image = ax.imshow(display, cmap="viridis", vmin=lo, vmax=hi,
                          origin="upper", extent=[0, COLS, ROWS, 0])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"Width ({COLS})", fontsize=8)
        ax.set_ylabel(f"Height ({ROWS})", fontsize=8)
        ax.tick_params(labelsize=7)
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.04)
        fig.colorbar(image, cax=cax).ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
