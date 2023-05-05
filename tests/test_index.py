import numpy as np
import pytest

from avalign.retrieval.index import EmbeddingIndex


def test_add_and_len():
    idx = EmbeddingIndex()
    idx.add(np.eye(4))
    assert len(idx) == 4
    assert idx.ids == ["0", "1", "2", "3"]


def test_custom_ids():
    idx = EmbeddingIndex()
    idx.add(np.eye(2), ids=["a", "b"])
    assert idx.ids == ["a", "b"]


def test_search_returns_self_first():
    vecs = np.eye(5)
    idx = EmbeddingIndex(metric="cosine")
    idx.add(vecs)
    indices, scores = idx.search(vecs, k=1)
    assert np.array_equal(indices[:, 0], np.arange(5))
    assert np.allclose(scores[:, 0], 1.0)


def test_search_k_capped_to_size():
    idx = EmbeddingIndex()
    idx.add(np.random.default_rng(0).standard_normal((3, 8)))
    indices, _ = idx.search(np.zeros((1, 8)), k=10)
    assert indices.shape[1] == 3


def test_empty_index_raises():
    with pytest.raises(ValueError):
        EmbeddingIndex().similarity(np.zeros((1, 4)))


def test_bad_metric_raises():
    with pytest.raises(ValueError):
        EmbeddingIndex(metric="l2")
