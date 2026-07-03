import numpy as np
import pytest

from avalign.align import similarity


def test_l2_normalize_unit_norm():
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = similarity.l2_normalize(x)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_cosine_matrix_identity():
    a = np.eye(4)
    sim = similarity.cosine_similarity_matrix(a, a)
    assert np.allclose(np.diag(sim), 1.0)
    off = sim - np.eye(4)
    assert np.allclose(off, 0.0, atol=1e-6)


def test_cosine_matrix_shape():
    a = np.random.default_rng(0).standard_normal((5, 8))
    b = np.random.default_rng(1).standard_normal((3, 8))
    assert similarity.cosine_similarity_matrix(a, b).shape == (5, 3)


def test_pairwise_cosine_matches_diag():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((6, 10))
    b = rng.standard_normal((6, 10))
    pair = similarity.pairwise_cosine(a, b)
    full = np.diag(similarity.cosine_similarity_matrix(a, b))
    assert np.allclose(pair, full)


def test_scale_logits_rejects_bad_temperature():
    with pytest.raises(ValueError):
        similarity.scale_logits(np.ones((2, 2)), temperature=0.0)


def test_dim_mismatch_raises():
    with pytest.raises(ValueError):
        similarity.cosine_similarity_matrix(np.ones((2, 3)), np.ones((2, 4)))
