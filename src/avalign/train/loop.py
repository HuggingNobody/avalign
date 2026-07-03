"""最小可用的训练循环：单步更新与多轮拟合。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from avalign.train.losses import clip_contrastive_loss

Batch = tuple[torch.Tensor, torch.Tensor]


def train_step(
    model: nn.Module,
    audio: torch.Tensor,
    video: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """一次前向 + 反向 + 参数更新，返回标量损失。"""
    model.train()
    optimizer.zero_grad()
    logits = model(audio, video)
    loss = clip_contrastive_loss(logits)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def fit(
    model: nn.Module,
    batches: Iterable[Batch],
    epochs: int = 1,
    lr: float = 1e-3,
) -> list[float]:
    """在若干 ``(audio, video)`` batch 上训练，返回每步损失历史。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    materialized = list(batches)
    for _ in range(epochs):
        for audio, video in materialized:
            history.append(train_step(model, audio, video, optimizer))
    return history
