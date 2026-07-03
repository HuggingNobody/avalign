"""双编码器对齐模型：音频编码器 + 视觉编码器 + 可学习温度。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from avalign.models.audio_encoder import AudioEncoder
from avalign.models.visual_encoder import VisualEncoder


class AVAlignModel(nn.Module):
    """把音频与视频编码到共享空间，并给出带温度缩放的相似度 logits。"""

    def __init__(
        self,
        audio_encoder: nn.Module | None = None,
        visual_encoder: nn.Module | None = None,
        embed_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder or AudioEncoder(embed_dim=embed_dim)
        self.visual_encoder = visual_encoder or VisualEncoder(embed_dim=embed_dim)
        # 参照 CLIP，用可学习的 log 温度并在前向时截断，避免数值爆炸
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.audio_encoder(audio), dim=-1)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual_encoder(video), dim=-1)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        a = self.encode_audio(audio)
        v = self.encode_video(video)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * a @ v.t()
