"""CLIP 风格的投影头：Linear → GELU → Linear，带残差与 LayerNorm。"""

from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """把编码器输出投影到共享对齐空间。"""

    def __init__(self, dim: int, proj_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, proj_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.fc(x)
        h = self.gelu(projected)
        h = self.fc2(h)
        h = self.dropout(h)
        h = h + projected  # 残差
        return self.norm(h)
