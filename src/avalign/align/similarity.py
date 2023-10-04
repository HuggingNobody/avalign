"""嵌入相似度工具：L2 归一化、余弦相似度矩阵与温度缩放。"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """按给定轴做 L2 归一化。"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组嵌入之间的余弦相似度矩阵。

    输入 ``a`` 形状 ``(m, d)``、``b`` 形状 ``(n, d)``，返回 ``(m, n)`` 矩阵，
    元素 ``[i, j]`` 是 ``a[i]`` 与 ``b[j]`` 的余弦相似度。
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("两组嵌入的维度必须一致")
    return l2_normalize(a, axis=1) @ l2_normalize(b, axis=1).T


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐行配对的余弦相似度，形状 ``(n,)``。"""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if a.shape != b.shape:
        raise ValueError("逐行配对要求两组嵌入形状一致")
    return np.sum(l2_normalize(a, axis=1) * l2_normalize(b, axis=1), axis=1)


def scale_logits(similarity: np.ndarray, temperature: float) -> np.ndarray:
    """用温度缩放相似度得到 logits：``sim / temperature``。"""
    if temperature <= 0:
        raise ValueError("temperature 必须为正数")
    return similarity / temperature
