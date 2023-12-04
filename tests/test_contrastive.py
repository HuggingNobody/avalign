import numpy as np

from avalign.align import contrastive


def test_softmax_sums_to_one():
    x = np.random.default_rng(0).standard_normal((4, 6))
    p = contrastive.softmax(x, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_log_softmax_matches_softmax():
    x = np.random.default_rng(1).standard_normal((3, 5))
    assert np.allclose(contrastive.log_softmax(x), np.log(contrastive.softmax(x)))


def test_info_nce_lower_when_aligned():
    rng = np.random.default_rng(2)
    emb = rng.standard_normal((16, 32))
    aligned = contrastive.info_nce(emb, emb.copy(), temperature=0.07)
    shuffled = contrastive.info_nce(emb, rng.permutation(emb, axis=0), temperature=0.07)
    assert aligned < shuffled


def test_clip_loss_symmetric_value():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((8, 16))
    v = rng.standard_normal((8, 16))
    loss = contrastive.clip_loss(a, v)
    assert loss > 0.0
    # perfectly aligned pairs give a smaller loss
    assert contrastive.clip_loss(a, a.copy()) < loss


def test_nt_xent_positive():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((10, 12))
    v = a + 0.01 * rng.standard_normal((10, 12))
    loss = contrastive.nt_xent(a, v, temperature=0.5)
    assert np.isfinite(loss)
    assert loss > 0.0


def test_cross_entropy_perfect_prediction_is_small():
    logits = np.eye(5) * 20.0
    labels = np.arange(5)
    loss = contrastive.cross_entropy(logits, labels)
    assert loss < 1e-3


def test_clip_loss_equals_manual_symmetric_average(synthetic_pairs):
    audio, video = synthetic_pairs
    from avalign.align.similarity import cosine_similarity_matrix

    logits = cosine_similarity_matrix(audio, video) / 0.07
    labels = np.arange(len(audio))
    manual = 0.5 * (
        contrastive.cross_entropy(logits, labels) + contrastive.cross_entropy(logits.T, labels)
    )
    assert np.isclose(contrastive.clip_loss(audio, video, temperature=0.07), manual)


def test_single_sample_batch_is_zero_loss():
    a = np.ones((1, 8))
    v = np.ones((1, 8))
    # 只有一个样本时正类就是全部，交叉熵应为 0
    assert contrastive.info_nce(a, v) < 1e-6


def test_lower_temperature_sharpens_loss(synthetic_pairs):
    audio, video = synthetic_pairs
    hot = contrastive.clip_loss(audio, video, temperature=0.2)
    cold = contrastive.clip_loss(audio, video, temperature=0.02)
    assert cold < hot
