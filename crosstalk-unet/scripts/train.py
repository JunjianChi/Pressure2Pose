"""Train the crosstalk-removal U-Net.

Input is [crosstalk image, mask]; the target is the diode reference. Loss and the
reported R2 are computed only over active sensor cells, so the empty array never
contributes.

    python scripts/train.py --data data/processed/pairs.npz --epochs 100 --out checkpoints/unet.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from crosstalk.baseline import FrameMLP
from crosstalk.data import CrosstalkDataset, temporal_split
from crosstalk.metrics import MaskedScore, masked_mse
from crosstalk.model import UNet

MODELS = {"unet": UNet, "mlp": FrameMLP}


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (masked MSE, R2) over a loader, active cells only."""
    model.eval()
    score = MaskedScore()
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        score.update(model(x), y, m)
    return score.mse, score.r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/pairs.npz")
    ap.add_argument("--model", choices=sorted(MODELS), default="unet")
    ap.add_argument("--out", default="checkpoints/unet.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CrosstalkDataset(args.data)
    train_idx, val_idx = temporal_split(len(ds), args.val_frac)
    train_ld = DataLoader(Subset(ds, train_idx), batch_size=args.batch, shuffle=True)
    val_ld = DataLoader(Subset(ds, val_idx), batch_size=args.batch)

    model = MODELS[args.model]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_r2 = -float("inf")
    history = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = n_batches = 0
        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = masked_mse(model(x), y, m)
            loss.backward()
            opt.step()
            train_loss += float(loss)
            n_batches += 1

        mse, r2 = evaluate(model, val_ld, device)
        history.append(
            {"epoch": epoch, "train_mse": train_loss / max(n_batches, 1),
             "val_mse": mse, "val_r2": r2}
        )
        print(f"epoch {epoch:3d}  val_mse {mse:.5f}  val_r2 {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), args.out)

    with open(args.out + ".history.json", "w") as fh:
        json.dump({"model": args.model, "seed": args.seed, "epochs": history}, fh)
    print(f"best val R2 {best_r2:.4f}  saved to {args.out}")


if __name__ == "__main__":
    main()
