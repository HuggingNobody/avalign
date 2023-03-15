"""把不同模态的特征投影到共享嵌入空间的线性头（纯 NumPy）。"""

from __future__ import annotations

import numpy as np

from avalign.align.similarity import l2_normalize


class LinearProjection:
    """带 Xavier 初始化的线性投影头。

    这是训练好的 ``torch`` 投影头的一个轻量替身，方便在没有深度学习框架
    的情况下跑通对齐/检索流程。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.weight = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.bias = np.zeros(out_dim) if bias else None

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def encode(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        """投影后可选做 L2 归一化，方便直接送入余弦相似度。"""
        out = self.forward(x)
        return l2_normalize(out, axis=1) if normalize else out
