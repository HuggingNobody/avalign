"""合成音视频数据，用于测试、示例与冒烟评估。"""

from __future__ import annotations

import numpy as np

from avalign.data.pairs import AVPairBatch

RngLike = int | np.random.Generator | None


def make_synthetic_embeddings(
    num_pairs: int,
    dim: int = 32,
    latent_dim: int = 8,
    noise: float = 0.1,
    seed: RngLike = None,
) -> AVPairBatch:
    """从共享隐变量生成配对的音频/视频嵌入。

    每一对共享一个隐向量 ``z``，两条模态编码器共享一个主投影，再各自叠加
    一点模态特有的扰动。这样对角线上的样本对天然更相似——正好用来检验
    对齐与检索是否工作。
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((num_pairs, latent_dim))
    w_shared = rng.standard_normal((latent_dim, dim))
    w_audio = w_shared + 0.2 * rng.standard_normal((latent_dim, dim))
    w_video = w_shared + 0.2 * rng.standard_normal((latent_dim, dim))
    audio = z @ w_audio + noise * rng.standard_normal((num_pairs, dim))
    video = z @ w_video + noise * rng.standard_normal((num_pairs, dim))
    ids = [str(i) for i in range(num_pairs)]
    return AVPairBatch(audio_embed=audio, video_embed=video, ids=ids)


def _shift(x: np.ndarray, offset: int) -> np.ndarray:
    out = np.zeros_like(x)
    if offset >= 0:
        out[offset:] = x[: len(x) - offset]
    else:
        out[:offset] = x[-offset:]
    return out


def make_synced_envelopes(
    length: int = 200,
    offset: int = 5,
    noise: float = 0.05,
    seed: RngLike = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """生成一对共享结构、但相差已知偏移的时间包络，返回 ``(audio, video, offset)``。"""
    rng = np.random.default_rng(seed)
    events = np.zeros(length)
    idx = rng.choice(length - 20, size=5, replace=False) + 10
    events[idx] = rng.uniform(1.0, 3.0, size=5)
    kernel = np.exp(-np.linspace(0.0, 3.0, 15))
    base = np.convolve(events, kernel, mode="same")

    audio_env = base + noise * rng.standard_normal(length)
    video_env = _shift(base, offset) + noise * rng.standard_normal(length)
    return audio_env, video_env, offset
