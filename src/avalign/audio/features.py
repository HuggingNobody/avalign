"""梅尔谱、对数梅尔谱与 MFCC。"""

from __future__ import annotations

import numpy as np

from avalign.audio.framing import power_spectrum


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """HTK 风格的 Hz→Mel 转换。"""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Mel→Hz 转换。"""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """构造三角梅尔滤波器组，形状 ``(n_mels, n_fft // 2 + 1)``。"""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        lo, center, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        left = (fft_freqs - lo) / (center - lo + 1e-10)
        right = (hi - fft_freqs) / (hi - center + 1e-10)
        fb[m] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


def mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float | None = None,
    window: str = "hann",
) -> np.ndarray:
    """梅尔谱，形状 ``(n_mels, num_frames)``。"""
    ps = power_spectrum(signal, n_fft, hop_length, window)
    fb = mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
    return fb @ ps


def log_mel_spectrogram(signal: np.ndarray, ref: float = 1e-6, **kwargs: object) -> np.ndarray:
    """对数梅尔谱 ``log(mel + ref)``。"""
    mel = mel_spectrogram(signal, **kwargs)  # type: ignore[arg-type]
    return np.log(mel + ref)


def _dct_ii_basis(n_out: int, n_in: int) -> np.ndarray:
    """正交归一化 DCT-II 基矩阵，形状 ``(n_out, n_in)``。"""
    k = np.arange(n_out)[:, None]
    m = np.arange(n_in)[None, :]
    basis = np.cos(np.pi * (m + 0.5) * k / n_in)
    basis[0] *= np.sqrt(1.0 / n_in)
    basis[1:] *= np.sqrt(2.0 / n_in)
    return basis


def mfcc(signal: np.ndarray, n_mfcc: int = 20, **kwargs: object) -> np.ndarray:
    """在对数梅尔谱上做 DCT，得到 MFCC，形状 ``(n_mfcc, num_frames)``。"""
    log_mel = log_mel_spectrogram(signal, **kwargs)
    n_mels = log_mel.shape[0]
    if n_mfcc > n_mels:
        raise ValueError("n_mfcc 不能超过 n_mels")
    basis = _dct_ii_basis(n_mfcc, n_mels)
    return basis @ log_mel
