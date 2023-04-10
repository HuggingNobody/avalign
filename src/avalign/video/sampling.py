"""从帧序列中采样关键帧：均匀采样与 TSN 式分段采样。"""

from __future__ import annotations

import numpy as np

RngLike = int | np.random.Generator | None


def uniform_frame_indices(num_available: int, num_sampled: int) -> np.ndarray:
    """在 ``[0, num_available)`` 上均匀取 ``num_sampled`` 个索引。"""
    if num_available <= 0 or num_sampled <= 0:
        raise ValueError("帧数与采样数必须为正")
    if num_sampled >= num_available:
        return np.arange(num_available)
    return np.linspace(0, num_available - 1, num_sampled).round().astype(int)


def segment_indices(num_available: int, num_segments: int, *, rng: RngLike = None) -> np.ndarray:
    """TSN 式分段：把序列切成若干段，每段随机取一帧。"""
    if num_available <= 0 or num_segments <= 0:
        raise ValueError("帧数与段数必须为正")
    generator = np.random.default_rng(rng)
    bounds = np.linspace(0, num_available, num_segments + 1).astype(int)
    out = []
    for i in range(num_segments):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        hi = max(hi, lo + 1)
        out.append(int(generator.integers(lo, min(hi, num_available))))
    return np.array(out)


def sample_frames(frames: np.ndarray, num_sampled: int) -> np.ndarray:
    """按均匀索引从帧堆叠中取子集。"""
    frames = np.asarray(frames)
    idx = uniform_frame_indices(frames.shape[0], num_sampled)
    return frames[idx]
