"""跨模态 Top-K 搜索与双向检索的便捷封装。"""

from __future__ import annotations

import numpy as np

from avalign.retrieval.index import EmbeddingIndex


def search_topk(
    query_embeds: np.ndarray,
    gallery_embeds: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """在 gallery 中为每个 query 检索 Top-K，返回 ``(indices, scores)``。"""
    index = EmbeddingIndex(metric=metric)
    index.add(gallery_embeds)
    return index.search(query_embeds, k=k)


def retrieve(
    query_embeds: np.ndarray,
    gallery_embeds: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> np.ndarray:
    """只返回 Top-K 的索引，形状 ``(num_queries, k)``。"""
    indices, _ = search_topk(query_embeds, gallery_embeds, k, metric)
    return indices


def bidirectional_retrieval(
    audio_embeds: np.ndarray,
    video_embeds: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """同时给出 音频→视频 与 视频→音频 两个方向的检索结果。"""
    a2v = retrieve(audio_embeds, video_embeds, k, metric)
    v2a = retrieve(video_embeds, audio_embeds, k, metric)
    return a2v, v2a
