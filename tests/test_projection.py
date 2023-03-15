import numpy as np

from avalign.align.projection import LinearProjection


def test_projection_output_shape():
    proj = LinearProjection(64, 16, seed=0)
    x = np.random.default_rng(0).standard_normal((5, 64))
    out = proj(x)
    assert out.shape == (5, 16)
    assert proj.out_dim == 16


def test_projection_is_deterministic_with_seed():
    a = LinearProjection(8, 4, seed=42)
    b = LinearProjection(8, 4, seed=42)
    assert np.allclose(a.weight, b.weight)


def test_encode_normalizes():
    proj = LinearProjection(32, 8, seed=1)
    x = np.random.default_rng(1).standard_normal((7, 32))
    out = proj.encode(x, normalize=True)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_no_bias_option():
    proj = LinearProjection(10, 3, bias=False, seed=2)
    assert proj.bias is None
    x = np.zeros((2, 10))
    assert np.allclose(proj(x), 0.0)
