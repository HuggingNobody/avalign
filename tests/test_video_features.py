import numpy as np

from avalign.video import features


def _random_frames(t=6, h=16, w=16, seed=0):
    return np.random.default_rng(seed).random((t, h, w, 3))


def test_to_grayscale_shape():
    frames = _random_frames()
    gray = features.to_grayscale(frames)
    assert gray.shape == (6, 16, 16)


def test_to_grayscale_rejects_non_rgb():
    import pytest

    with pytest.raises(ValueError):
        features.to_grayscale(np.zeros((4, 4, 2)))


def test_color_histogram_normalized():
    frame = _random_frames(t=1)[0]
    hist = features.color_histogram(frame, bins=8)
    assert hist.shape == (24,)
    assert np.isclose(hist.sum(), 1.0)


def test_motion_energy_length():
    frames = _random_frames(t=10)
    energy = features.motion_energy(frames)
    assert energy.shape == (9,)
    assert np.all(energy >= 0.0)


def test_motion_energy_zero_for_static_clip():
    frames = np.ones((5, 8, 8, 3))
    assert np.allclose(features.motion_energy(frames), 0.0)


def test_grid_pool_dim():
    frame = _random_frames(t=1)[0]
    pooled = features.grid_pool(frame, grid=4)
    assert pooled.shape == (16,)


def test_frame_embedding_stack():
    frames = _random_frames(t=7)
    emb = features.frame_embedding(frames, bins=8)
    assert emb.shape == (7, 24)
