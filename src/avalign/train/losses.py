"""PyTorch 版本的对比损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """CLIP 对称交叉熵，``logits`` 形状 ``(B, B)``，正样本在对角线。"""
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)
    loss_a2v = F.cross_entropy(logits, labels)
    loss_v2a = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a2v + loss_v2a)


def info_nce_loss(
    audio: torch.Tensor, video: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """单向 InfoNCE：先归一化再算带温度的相似度。"""
    audio = F.normalize(audio, dim=-1)
    video = F.normalize(video, dim=-1)
    logits = audio @ video.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
