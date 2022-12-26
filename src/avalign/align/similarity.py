"""嵌入相似度工具。

TODO: L2 归一化、余弦相似度矩阵与温度缩放。
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """按给定轴做 L2 归一化。"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)
