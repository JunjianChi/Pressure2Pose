"""U-Net that maps a crosstalk-contaminated pressure image to a crosstalk-free one.

Input is (B, 2, 33, 15): channel 0 is the normalized crossbar reading, channel 1 is
the active-sensor mask. Output is (B, 1, 33, 15) in [0, 1] via a final sigmoid.

`depth` sets how many times the encoder halves the image and the channel width
doubles, so the network is one number wide rather than a hand-written stack of
levels. 33x15 is not divisible by 2**depth, so `forward` pads the input up to the
next multiple once and crops the result back at the end; every shape in between is
exact and no skip connection needs patching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv - BatchNorm - ReLU, twice."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Encoder-decoder with skip connections, `depth` levels deep."""

    def __init__(self, in_ch: int = 2, out_ch: int = 1, base: int = 32, depth: int = 2):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        widths = [base * 2**i for i in range(depth + 1)]

        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        ch = in_ch
        for width in widths[:-1]:
            self.encoders.append(DoubleConv(ch, width))
            ch = width

        self.bottleneck = DoubleConv(ch, widths[-1])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = widths[-1]
        for width in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch, width, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width))
            ch = width

        self.head = nn.Conv2d(ch, out_ch, kernel_size=1)

    def forward(self, x):
        rows, cols = x.shape[-2:]
        x = self._pad_to_multiple(x, 2**self.depth)

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat([up(x), skip], dim=1))

        return torch.sigmoid(self.head(x))[..., :rows, :cols]

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
        """Extend the bottom and right edges so both sides divide by `m`."""
        rows, cols = x.shape[-2:]
        return F.pad(x, (0, -cols % m, 0, -rows % m))
