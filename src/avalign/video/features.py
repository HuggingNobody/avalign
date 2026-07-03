"""简单的视觉特征：灰度、颜色直方图、运动能量与网格池化嵌入。

这些特征都很轻量，作为预训练视觉编码器的替身，用来在纯 NumPy 环境下
跑通对齐与检索流程。
"""

from __future__ import annotations

import numpy as np

_LUMA = np.array([0.299, 0.587, 0.114])


def to_grayscale(frames: np.ndarray) -> np.ndarray:
    """把 RGB 帧转成灰度，支持 ``(..., H, W, 3)``。"""
    frames = np.asarray(frames, dtype=float)
    if frames.shape[-1] != 3:
        raise ValueError("最后一维必须是 RGB 三通道")
    return frames @ _LUMA


def color_histogram(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    """每个颜色通道的归一化直方图，拼接成长度 ``3 * bins`` 的向量。"""
    frame = np.asarray(frame, dtype=float)
    if frame.max(initial=0.0) > 1.0:
        frame = frame / 255.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    parts = [np.histogram(frame[..., c], bins=edges)[0] for c in range(frame.shape[-1])]
    hist = np.concatenate(parts).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def motion_energy(frames: np.ndarray) -> np.ndarray:
    """相邻帧差的平均幅度，返回长度 ``T - 1`` 的运动能量曲线。

    这条曲线常被用作视觉端的时间包络，配合音频包络做同步偏移估计。
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim == 4:
        frames = to_grayscale(frames)
    if frames.shape[0] < 2:
        return np.zeros(0)
    diffs = np.abs(np.diff(frames, axis=0))
    return diffs.reshape(diffs.shape[0], -1).mean(axis=1)


def grid_pool(frame: np.ndarray, grid: int = 4) -> np.ndarray:
    """把单帧灰度图平均池化成 ``grid x grid`` 网格并展平。"""
    frame = np.asarray(frame, dtype=float)
    if frame.ndim == 3:
        frame = to_grayscale(frame)
    h, w = frame.shape
    hh, ww = (h // grid) * grid, (w // grid) * grid
    frame = frame[:hh, :ww]
    pooled = frame.reshape(grid, hh // grid, grid, ww // grid).mean(axis=(1, 3))
    return pooled.reshape(-1)


def frame_embedding(frames: np.ndarray, bins: int = 8) -> np.ndarray:
    """把每帧转成颜色直方图，堆叠成 ``(T, 3 * bins)`` 的帧级嵌入。"""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim == 3:
        frames = frames[None]
    return np.stack([color_histogram(f, bins=bins) for f in frames], axis=0)
