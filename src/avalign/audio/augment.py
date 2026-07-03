"""谱图数据增强：时间/频率掩码与加性噪声。"""

from __future__ import annotations

import numpy as np

RngLike = int | np.random.Generator | None


def add_noise(signal: np.ndarray, snr_db: float, rng: RngLike = None) -> np.ndarray:
    """按目标信噪比添加高斯白噪声。"""
    generator = np.random.default_rng(rng)
    signal = np.asarray(signal, dtype=float)
    sig_power = float(np.mean(signal**2))
    if sig_power == 0.0:
        return signal.copy()
    noise = generator.standard_normal(signal.shape)
    noise_power = float(np.mean(noise**2))
    factor = np.sqrt(sig_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return signal + factor * noise


def spec_augment(
    spec: np.ndarray,
    *,
    num_freq_masks: int = 1,
    freq_mask_width: int = 8,
    num_time_masks: int = 1,
    time_mask_width: int = 16,
    rng: RngLike = None,
) -> np.ndarray:
    """SpecAugment 风格的时频掩码，输入形状 ``(n_freq, n_time)``。"""
    generator = np.random.default_rng(rng)
    out = np.asarray(spec, dtype=float).copy()
    n_freq, n_time = out.shape

    for _ in range(num_freq_masks):
        width = int(generator.integers(0, min(freq_mask_width, n_freq) + 1))
        if width > 0:
            start = int(generator.integers(0, n_freq - width + 1))
            out[start : start + width, :] = 0.0

    for _ in range(num_time_masks):
        width = int(generator.integers(0, min(time_mask_width, n_time) + 1))
        if width > 0:
            start = int(generator.integers(0, n_time - width + 1))
            out[:, start : start + width] = 0.0

    return out
