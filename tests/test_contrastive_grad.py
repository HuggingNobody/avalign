import numpy as np

from avalign.align import contrastive


def _numeric_grad(logits, labels, eps=1e-6):
    grad = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            plus = logits.copy()
            plus[i, j] += eps
            minus = logits.copy()
            minus[i, j] -= eps
            lp = contrastive.cross_entropy(plus, labels)
            lm = contrastive.cross_entropy(minus, labels)
            grad[i, j] = (lp - lm) / (2 * eps)
    return grad


def test_cross_entropy_analytic_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 5))
    labels = rng.integers(0, 5, size=4)
    loss, grad = contrastive.cross_entropy(logits, labels, return_grad=True)
    numeric = _numeric_grad(logits, labels)
    assert np.isfinite(loss)
    assert np.allclose(grad, numeric, atol=1e-5)


def test_gradient_rows_sum_to_zero():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((6, 4))
    labels = rng.integers(0, 4, size=6)
    _, grad = contrastive.cross_entropy(logits, labels, return_grad=True)
    # softmax 概率减 one-hot，逐行和应为 0
    assert np.allclose(grad.sum(axis=1), 0.0, atol=1e-8)
