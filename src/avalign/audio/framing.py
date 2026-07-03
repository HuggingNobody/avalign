"""信号分帧与短时傅里叶变换。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def hann_window(length: int) -> np.ndarray:
    """周期性 Hann 窗。"""
    if length <= 1:
        return np.ones(max(length, 0), dtype=float)
    n = np.arange(length)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))


def hamming_window(length: int) -> np.ndarray:
    """Hamming 窗。"""
    if length <= 1:
        return np.ones(max(length, 0), dtype=float)
    n = np.arange(length)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))


_WINDOWS: dict[str, Callable[[int], np.ndarray]] = {
    "hann": hann_window,
    "hamming": hamming_window,
}


def get_window(name: str, length: int) -> np.ndarray:
    """按名称取窗函数。"""
    try:
        fn = _WINDOWS[name]
    except KeyError as exc:
        raise ValueError(f"未知的窗函数: {name}") from exc
    return fn(length)


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    *,
    center: bool = False,
) -> np.ndarray:
    """把一维信号切成互相重叠的帧，返回 ``(num_frames, frame_length)``。"""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("frame_signal 只接受一维信号")
    if signal.size == 0:
        raise ValueError("signal 不能为空")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length 与 hop_length 必须为正数")
    if center:
        signal = np.pad(signal, frame_length // 2, mode="reflect")
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(num_frames)[:, None]
    return signal[idx]


def stft(
    signal: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
    *,
    center: bool = True,
) -> np.ndarray:
    """短时傅里叶变换，返回复数谱 ``(n_fft // 2 + 1, num_frames)``。"""
    frames = frame_signal(signal, n_fft, hop_length, center=center)
    win = get_window(window, n_fft)
    spec = np.fft.rfft(frames * win[None, :], n=n_fft, axis=1)
    return spec.T


def power_spectrum(
    signal: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
    *,
    center: bool = True,
) -> np.ndarray:
    """功率谱 ``|STFT|**2``，形状 ``(n_fft // 2 + 1, num_frames)``。"""
    spec = stft(signal, n_fft, hop_length, window, center=center)
    return np.abs(spec) ** 2
