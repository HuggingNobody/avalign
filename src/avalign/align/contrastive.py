"""对比学习损失：softmax、交叉熵、InfoNCE、CLIP 与 NT-Xent。"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """沿给定轴的 softmax。"""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值上等价于 ``log(softmax(x))``。"""
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    return_grad: bool = False,
) -> float | tuple[float, np.ndarray]:
    """行方向的多分类交叉熵，``targets`` 是每行正类的列索引。"""
    logits = np.atleast_2d(logits)
    n = logits.shape[0]
    probs = softmax(logits, axis=-1)
    picked = probs[np.arange(n), targets]
    loss = float(-np.mean(np.log(picked + 1e-12)))
    if not return_grad:
        return loss
    grad = probs.copy()
    grad[np.arange(n), targets] -= 1.0
    grad /= n
    return loss, grad
