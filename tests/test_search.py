import numpy as np

from avalign.retrieval import search


def test_search_topk_shapes():
    q = np.random.default_rng(0).standard_normal((4, 16))
    g = np.random.default_rng(1).standard_normal((20, 16))
    idx, scores = search.search_topk(q, g, k=5)
    assert idx.shape == (4, 5)
    assert scores.shape == (4, 5)


def test_retrieve_finds_exact_match():
    gallery = np.eye(6)
    queries = np.eye(6)
    idx = search.retrieve(queries, gallery, k=1)
    assert np.array_equal(idx[:, 0], np.arange(6))


def test_bidirectional_shapes():
    a = np.random.default_rng(2).standard_normal((5, 8))
    v = np.random.default_rng(3).standard_normal((5, 8))
    a2v, v2a = search.bidirectional_retrieval(a, v, k=3)
    assert a2v.shape == (5, 3)
    assert v2a.shape == (5, 3)


def test_scores_descending():
    q = np.random.default_rng(4).standard_normal((3, 10))
    g = np.random.default_rng(5).standard_normal((30, 10))
    _, scores = search.search_topk(q, g, k=8)
    assert np.all(np.diff(scores, axis=1) <= 1e-9)
