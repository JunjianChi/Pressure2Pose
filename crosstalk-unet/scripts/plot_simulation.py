"""Show the simulator's sneak-path effect: true resistance map vs crossbar readout.

    python scripts/plot_simulation.py --out simulation.png
"""

from __future__ import annotations

import argparse

import numpy as np

from crosstalk.sensor import COLS, ROWS, active_mask
from crosstalk.simulate import CrossbarReadout

R_UNLOADED = 20000.0
R_PRESSED = 300.0


def pressed_pattern() -> np.ndarray:
    """A heel and a forefoot press on an otherwise unloaded grid."""
    from scipy.ndimage import gaussian_filter

    weight = np.zeros((ROWS, COLS))
    weight[26, 7] = 1.0
    weight[8, 6] = 0.8
    weight = gaussian_filter(weight, sigma=2.2)
    weight /= weight.max()
    return R_UNLOADED - (R_UNLOADED - R_PRESSED) * weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="simulation.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    truth = pressed_pattern()
    measured = CrossbarReadout().measure(truth)
    mask = active_mask()

    # Each panel is scaled to its own robust range: the readout spans only a few
    # percent of the true scale, and a shared scale would flatten it to one colour.
    panels = [
        ("true resistance (ohm)", truth),
        ("crossbar readout (ohm)", measured),
        ("readout / true", measured / truth),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.4))
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
