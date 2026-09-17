from __future__ import annotations
import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(
            channels, channels, 3, padding=self.pad, dilation=dilation
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.conv(x)[..., : -self.pad])


class ForecastNet(nn.Module):
    def __init__(self, features, bands, horizons, hidden=32, mode="fusion"):
        super().__init__()
        self.mode = mode
        if mode not in {"tabular", "image", "fusion"}:
            raise ValueError(mode)
        if mode != "image":
            self.tabular = nn.Sequential(
                nn.Conv1d(features, hidden, 1),
                nn.ReLU(),
                CausalBlock(hidden, 1),
                CausalBlock(hidden, 2),
            )
        if mode != "tabular":
            self.spatial = nn.Sequential(
                nn.Conv2d(bands * 2, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, hidden, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(hidden * 4, hidden),
                nn.ReLU(),
            )
            self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 if mode == "fusion" else hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizons),
        )

    def forward(self, tabular, images):
        parts = []
        if self.mode != "image":
            parts.append(self.tabular(tabular.transpose(1, 2))[..., -1])
        if self.mode != "tabular":
            b, l, c, h, w = images.shape
            spatial = self.spatial(images.reshape(b * l, c, h, w)).reshape(b, l, -1)
            temporal, _ = self.temporal(spatial)
            parts.append(temporal[:, -1])
        return self.head(torch.cat(parts, dim=-1))
