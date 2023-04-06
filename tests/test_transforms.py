import numpy as np

from avalign.video import transforms


def test_resize_downscale():
    frame = np.random.default_rng(0).random((32, 32, 3))
    out = transforms.resize(frame, 8)
    assert out.shape == (8, 8, 3)


def test_resize_grayscale():
    frame = np.random.default_rng(1).random((20, 40))
    out = transforms.resize(frame, (10, 10))
    assert out.shape == (10, 10)


def test_center_crop():
    frame = np.arange(64).reshape(8, 8)
    out = transforms.center_crop(frame, 4)
    assert out.shape == (4, 4)
    assert out[0, 0] == frame[2, 2]


def test_normalize_scales_uint_range():
    frame = np.full((4, 4, 3), 255.0)
    out = transforms.normalize(frame)
    assert np.allclose(out, 1.0)


def test_normalize_with_mean_std():
    frame = np.full((2, 2), 0.5)
    out = transforms.normalize(frame, mean=0.5, std=0.5)
    assert np.allclose(out, 0.0)
