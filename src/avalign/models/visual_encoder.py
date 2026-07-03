"""对视频帧做 2D 卷积、再沿时间维平均池化的视觉编码器。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class VisualEncoder(nn.Module):
    """编码单帧 ``(B, C, H, W)`` 或帧序列 ``(B, T, C, H, W)``。

    对帧序列会逐帧编码后在时间维取平均，得到 ``(B, embed_dim)``。
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        channels: Sequence[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = in_channels
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

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # (B, T, C, H, W)
            b, t = x.shape[:2]
            flat = x.reshape(b * t, *x.shape[2:])
            return self._encode(flat).reshape(b, t, -1).mean(dim=1)
        return self._encode(x)
