"""共享测试夹具。"""

import numpy as np
import pytest

from avalign.data.synthetic import make_synthetic_embeddings


@pytest.fixture
def rng():
    return np.random.default_rng(20240601)


@pytest.fixture
def synthetic_pairs():
    """一批低噪声的合成音视频嵌入，返回 ``(audio, video)``。"""
    batch = make_synthetic_embeddings(50, dim=32, noise=0.08, seed=7)
    return batch.audio_embed, batch.video_embed
