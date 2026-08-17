"""Per-sensor test error map and prediction-reference scatter for a checkpoint.

    python scripts/plot_error_analysis.py --data data/processed/lab_test.npz \\
        --weights checkpoints/unet_lab.pt --out error_analysis.png
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from crosstalk.baseline import FrameMLP
from crosstalk.data import CrosstalkDataset
from crosstalk.model import UNet
from crosstalk.sensor import COLS, ROWS, active_mask

MODELS = {"unet": UNet, "mlp": FrameMLP}


@torch.no_grad()
def predict(model, ds, device, batch: int = 256) -> np.ndarray:
    preds = []
    for start in range(0, len(ds), batch):
        xs = np.stack([ds[i][0] for i in range(start, min(start + batch, len(ds)))])
        preds.append(model(torch.from_numpy(xs).to(device)).cpu().numpy()[:, 0])
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/lab_test.npz")
    ap.add_argument("--model", choices=sorted(MODELS), default="unet")
    ap.add_argument("--weights", default="checkpoints/unet_lab.pt")
    ap.add_argument("--out", default="error_analysis.png")
    ap.add_argument("--scatter-points", type=int, default=20000)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[args.model]().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    ds = CrosstalkDataset(args.data)
    pred = predict(model, ds, device)
    mask = active_mask()

    mae = np.abs(pred - ds.ref).mean(axis=0)
    mae_display = np.where(mask, mae, np.nan)

    rng = np.random.default_rng(0)
    p_flat = pred[:, mask].ravel()
    r_flat = ds.ref[:, mask].ravel()
    pick = rng.choice(len(p_flat), size=min(args.scatter_points, len(p_flat)), replace=False)

    fig, (ax_map, ax_sc) = plt.subplots(
        1, 2, figsize=(8.2, 4.0), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    image = ax_map.imshow(mae_display, cmap="viridis", origin="upper", extent=[0, COLS, ROWS, 0])
    ax_map.set_title(f"per-sensor test MAE (n={len(ds)})", fontsize=10)
    ax_map.set_xlabel(f"Width ({COLS})", fontsize=8)
    ax_map.set_ylabel(f"Height ({ROWS})", fontsize=8)
    ax_map.tick_params(labelsize=7)
    cax = make_axes_locatable(ax_map).append_axes("right", size="5%", pad=0.04)
    fig.colorbar(image, cax=cax).ax.tick_params(labelsize=7)

    ax_sc.plot([0, 1], [0, 1], color="gray", linewidth=1, zorder=1)
    ax_sc.scatter(r_flat[pick], p_flat[pick], s=2, alpha=0.05, zorder=2)
    mse = float(((p_flat - r_flat) ** 2).mean())
    r2 = 1.0 - mse / float(r_flat.var())
    ax_sc.set_title(f"prediction vs reference (R² = {r2:.3f})", fontsize=10)
    ax_sc.set_xlabel("reference pressed fraction", fontsize=8)
    ax_sc.set_ylabel("predicted pressed fraction", fontsize=8)
    ax_sc.set_xlim(0, 1)
    ax_sc.set_ylim(0, 1)
    ax_sc.set_aspect("equal")
    ax_sc.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
