"""端到端对齐流程：把两路特征投影到共享空间，再算损失/相似度/检索指标。"""

from __future__ import annotations

import numpy as np

from avalign.align.contrastive import clip_loss, info_nce, nt_xent
from avalign.align.projection import LinearProjection
from avalign.align.similarity import cosine_similarity_matrix, l2_normalize
from avalign.retrieval.metrics import retrieval_report

_LOSSES = {"clip": clip_loss, "info_nce": info_nce, "nt_xent": nt_xent}


class AVAlignPipeline:
    """一个不需要训练的对齐流程，用随机投影充当两路编码器。

    它把音频/视频特征分别投影并归一化到共享空间，然后暴露统一的接口做
    对比损失、相似度矩阵与检索评估——方便快速验证数据管线是否打通。
    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        embed_dim: int = 256,
        temperature: float = 0.07,
        loss: str = "clip",
        seed: int | None = None,
    ) -> None:
        if loss not in _LOSSES:
            raise ValueError(f"未知的损失类型: {loss}")
        self.audio_proj = LinearProjection(audio_dim, embed_dim, seed=seed)
        self.video_proj = LinearProjection(
            video_dim, embed_dim, seed=None if seed is None else seed + 1
        )
        self.temperature = temperature
        self.loss_name = loss

    def encode_audio(self, feats: np.ndarray) -> np.ndarray:
        return self.audio_proj.encode(feats, normalize=True)

    def encode_video(self, feats: np.ndarray) -> np.ndarray:
        return self.video_proj.encode(feats, normalize=True)

    def similarity(self, audio_feats: np.ndarray, video_feats: np.ndarray) -> np.ndarray:
        return cosine_similarity_matrix(
            self.encode_audio(audio_feats), self.encode_video(video_feats)
        )

    def loss(self, audio_feats: np.ndarray, video_feats: np.ndarray) -> float:
        a = self.encode_audio(audio_feats)
        v = self.encode_video(video_feats)
        return _LOSSES[self.loss_name](a, v, temperature=self.temperature)

    def evaluate(
        self,
        audio_feats: np.ndarray,
        video_feats: np.ndarray,
        ks: tuple[int, ...] = (1, 5, 10),
    ) -> dict[str, float]:
        sim = self.similarity(audio_feats, video_feats)
        return retrieval_report(sim, ks=ks)
