"""在对数梅尔谱上做 2D 卷积的轻量音频编码器。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class AudioEncoder(nn.Module):
    """把 ``(B, n_mels, T)`` 的对数梅尔谱编码成 ``(B, embed_dim)`` 的向量。"""

    def __init__(
        self,
        n_mels: int = 64,
        embed_dim: int = 256,
        channels: Sequence[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        layers: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(int(channels[-1]), embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (B, n_mels, T) -> 增加通道维
            x = x.unsqueeze(1)
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)
