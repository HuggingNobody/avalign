import numpy as np

from avalign.data import synthetic
from avalign.retrieval.metrics import recall_at_k
from avalign.align.similarity import cosine_similarity_matrix
from avalign.sync.offset import estimate_offset


def test_synthetic_embeddings_shapes():
    batch = synthetic.make_synthetic_embeddings(20, dim=16, seed=0)
    assert batch.audio_embed.shape == (20, 16)
    assert batch.video_embed.shape == (20, 16)
    assert batch.size == 20


def test_synthetic_pairs_are_retrievable():
    batch = synthetic.make_synthetic_embeddings(30, dim=32, noise=0.05, seed=1)
    sim = cosine_similarity_matrix(batch.audio_embed, batch.video_embed)
    # 低噪声下配对信号应明显强于随机，Recall@5 远高于随机水平
    assert recall_at_k(sim, k=5) > 0.5


def test_synced_envelopes_recover_offset():
    audio, video, true_offset = synthetic.make_synced_envelopes(
        length=300, offset=7, noise=0.02, seed=2
    )
    est = estimate_offset(video, audio, max_lag=30)
    assert abs(est - true_offset) <= 1
