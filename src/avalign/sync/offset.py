"""基于互相关的音视频时间偏移估计。"""

from __future__ import annotations

import numpy as np


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """两条一维包络的归一化互相关。

    返回 ``(lags, corr)``，其中 ``corr[i]`` 是把 ``x`` 相对 ``y`` 平移
    ``lags[i]`` 个采样点后的归一化相关系数。
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    nx, ny = len(x), len(y)
    if max_lag is None:
        max_lag = max(nx, ny) - 1
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y)) + 1e-12

    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(len(lags), dtype=float)
    for i, lag in enumerate(lags):
        if lag >= 0:
            a, b = x[lag:], y[: nx - lag]
        else:
            a, b = x[: ny + lag], y[-lag:]
        m = min(len(a), len(b))
        corr[i] = float(np.sum(a[:m] * b[:m]) / denom) if m > 0 else 0.0
    return lags, corr


def estimate_offset(x: np.ndarray, y: np.ndarray, max_lag: int | None = None) -> int:
    """估计把 ``x`` 对齐到 ``y`` 所需的滞后（正值表示 ``x`` 落后于 ``y``）。"""
    lags, corr = cross_correlation(x, y, max_lag)
    return int(lags[int(np.argmax(corr))])


def align_by_offset(
    x: np.ndarray,
    y: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按给定偏移裁剪出两条序列的重叠部分。"""
    x = np.asarray(x)
    y = np.asarray(y)
    if offset >= 0:
        xa, ya = x[offset:], y[: len(x) - offset]
    else:
        xa, ya = x[: len(y) + offset], y[-offset:]
    m = min(len(xa), len(ya))
    return xa[:m], ya[:m]
