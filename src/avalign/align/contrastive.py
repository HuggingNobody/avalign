"""对比学习损失：softmax、交叉熵、InfoNCE、CLIP 与 NT-Xent。"""

from __future__ import annotations

import numpy as np

from avalign.align.similarity import cosine_similarity_matrix, l2_normalize


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


def info_nce(query: np.ndarray, key: np.ndarray, temperature: float = 0.07) -> float:
    """单向 InfoNCE 损失，配对样本位于相似度矩阵的对角线。"""
    sim = cosine_similarity_matrix(query, key)
    logits = sim / temperature
    labels = np.arange(logits.shape[0])
    return cross_entropy(logits, labels)  # type: ignore[return-value]


def clip_loss(audio: np.ndarray, video: np.ndarray, temperature: float = 0.07) -> float:
    """CLIP 风格的对称交叉熵：音频→视频 与 视频→音频 取平均。"""
    sim = cosine_similarity_matrix(audio, video)
    logits = sim / temperature
    labels = np.arange(logits.shape[0])
    loss_a2v = cross_entropy(logits, labels)
    loss_v2a = cross_entropy(logits.T, labels)
    return 0.5 * (loss_a2v + loss_v2a)  # type: ignore[operator]
