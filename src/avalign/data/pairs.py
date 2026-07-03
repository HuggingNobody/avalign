"""音视频样本对的轻量数据容器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AVPair:
    """单条音视频样本对。

    ``audio`` 可以是波形或已抽取的特征；``video`` 通常是形如
    ``(T, H, W, C)`` 的帧堆叠，也可以是预抽取的帧特征。
    """

    audio: np.ndarray
    video: np.ndarray
    label: int | None = None
    pair_id: str | None = None


@dataclass
class AVPairBatch:
    """一个 batch 的音视频嵌入，配对索引沿 0 轴对齐。"""

    audio_embed: np.ndarray
    video_embed: np.ndarray
    ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.audio_embed.shape[0] != self.video_embed.shape[0]:
            raise ValueError("音频与视频嵌入的样本数必须一致")

    @property
    def size(self) -> int:
        """batch 中的样本对数量。"""
        return int(self.audio_embed.shape[0])
