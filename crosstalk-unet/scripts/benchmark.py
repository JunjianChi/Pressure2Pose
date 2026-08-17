"""Single-frame inference latency of each model on this machine's CPU/GPU.

The insole streams frames continuously, so the correction budget is per frame.

    python scripts/benchmark.py --iters 200
"""

from __future__ import annotations

import argparse
import time

import torch

from crosstalk.baseline import FrameMLP
from crosstalk.model import UNet
from crosstalk.sensor import COLS, ROWS

MODELS = {"unet": UNet, "mlp": FrameMLP}


@torch.no_grad()
def benchmark(model, device, iters: int, warmup: int = 20) -> dict:
    model.eval()
    x = torch.rand(1, 2, ROWS, COLS, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "mean_ms": sum(times) / len(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "fps": 1000.0 * len(times) / sum(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads; pinned so a quoted latency is reproducible")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, cls in MODELS.items():
        model = cls().to(device)
        params = sum(p.numel() for p in model.parameters())
        r = benchmark(model, device, args.iters)
        print(
            f"{name:5s} ({params:,} params, {device.type}): "
            f"mean {r['mean_ms']:.2f} ms  p50 {r['p50_ms']:.2f}  "
            f"p95 {r['p95_ms']:.2f}  {r['fps']:.0f} fps"
        )


if __name__ == "__main__":
    main()
