import numpy as np

from avalign.pipeline import AVAlignPipeline


def test_pipeline_encodes_to_unit_norm():
    pipe = AVAlignPipeline(audio_dim=24, video_dim=24, embed_dim=16, seed=0)
    feats = np.random.default_rng(0).standard_normal((5, 24))
    enc = pipe.encode_audio(feats)
    assert enc.shape == (5, 16)
    assert np.allclose(np.linalg.norm(enc, axis=1), 1.0)


def test_pipeline_similarity_shape():
    pipe = AVAlignPipeline(audio_dim=12, video_dim=20, embed_dim=8, seed=1)
    a = np.random.default_rng(1).standard_normal((6, 12))
    v = np.random.default_rng(2).standard_normal((4, 20))
    sim = pipe.similarity(a, v)
    assert sim.shape == (6, 4)


def test_pipeline_loss_is_finite():
    pipe = AVAlignPipeline(audio_dim=16, video_dim=16, embed_dim=8, loss="clip", seed=3)
    a = np.random.default_rng(3).standard_normal((10, 16))
    v = np.random.default_rng(4).standard_normal((10, 16))
    loss = pipe.loss(a, v)
    assert np.isfinite(loss)


def test_pipeline_evaluate_report():
    pipe = AVAlignPipeline(audio_dim=16, video_dim=16, embed_dim=16, seed=5)
    feats = np.random.default_rng(5).standard_normal((8, 16))
    report = pipe.evaluate(feats, feats, ks=(1, 5))
    assert "R@1" in report and "mAP" in report


def test_pipeline_rejects_unknown_loss():
    import pytest

    with pytest.raises(ValueError):
        AVAlignPipeline(audio_dim=4, video_dim=4, loss="triplet")
