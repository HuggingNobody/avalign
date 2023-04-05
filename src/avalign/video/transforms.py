"""帧的几何与数值变换：缩放、中心裁剪与归一化。"""

from __future__ import annotations

import numpy as np


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return size


def resize(frame: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """最近邻缩放，支持 ``(H, W)`` 或 ``(H, W, C)``。"""
    frame = np.asarray(frame)
    out_h, out_w = _as_hw(size)
    h, w = frame.shape[:2]
    ys = np.clip((np.arange(out_h) * (h / out_h)).astype(int), 0, h - 1)
    xs = np.clip((np.arange(out_w) * (w / out_w)).astype(int), 0, w - 1)
    return frame[ys][:, xs]


def center_crop(frame: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """从帧中心裁剪出 ``size`` 大小的区域。"""
    frame = np.asarray(frame)
    ch, cw = _as_hw(size)
    h, w = frame.shape[:2]
    top = max((h - ch) // 2, 0)
    left = max((w - cw) // 2, 0)
    return frame[top : top + ch, left : left + cw]


def normalize(frame: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """把像素缩放到 ``[0, 1]`` 后做 ``(x - mean) / std`` 标准化。"""
    frame = np.asarray(frame, dtype=float)
    if frame.max(initial=0.0) > 1.0:
        frame = frame / 255.0
    return (frame - mean) / std
