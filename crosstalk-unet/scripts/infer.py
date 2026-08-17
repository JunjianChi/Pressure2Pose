"""Run a trained U-Net on prepared pairs and report agreement with the diode reference.

    python scripts/infer.py --data data/processed/lab_test.npz --weights checkpoints/unet_lab.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from crosstalk.baseline import FrameMLP
from crosstalk.data import CrosstalkDataset
from crosstalk.metrics import MaskedScore
from crosstalk.model import UNet

MODELS = {"unet": UNet, "mlp": FrameMLP}


@torch.no_grad()
def run(model, ds, device, batch: int = 256):
    """Return predictions (N, 33, 15), computed in batches."""
    preds = []
    for start in range(0, len(ds), batch):
        xs = np.stack([ds[i][0] for i in range(start, min(start + batch, len(ds)))])
        preds.append(model(torch.from_numpy(xs).to(device)).cpu().numpy()[:, 0])
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/pairs.npz")
    ap.add_argument("--model", choices=sorted(MODELS), default="unet")
    ap.add_argument("--weights", default="checkpoints/unet_lab.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[args.model]().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    ds = CrosstalkDataset(args.data)
    pred = run(model, ds, device)

    score = MaskedScore()
    score.update(pred, ds.ref, np.broadcast_to(ds.mask, pred.shape))
    print(f"n={len(ds)}  masked_mse={score.mse:.5f}  r2={score.r2:.4f}")


if __name__ == "__main__":
    main()
