import numpy as np
import pytest

from avalign.video import sampling


def test_uniform_indices_span_range():
    idx = sampling.uniform_frame_indices(100, 8)
    assert idx.shape == (8,)
    assert idx[0] == 0
    assert idx[-1] == 99
    assert np.all(np.diff(idx) > 0)


def test_uniform_indices_when_oversampled():
    idx = sampling.uniform_frame_indices(5, 10)
    assert np.array_equal(idx, np.arange(5))


def test_segment_indices_are_monotone_and_in_bounds():
    idx = sampling.segment_indices(40, 8, rng=0)
    assert idx.shape == (8,)
    assert np.all(idx >= 0)
    assert np.all(idx < 40)


def test_sample_frames_subset():
    frames = np.arange(20 * 4 * 4 * 3).reshape(20, 4, 4, 3)
    out = sampling.sample_frames(frames, 5)
    assert out.shape == (5, 4, 4, 3)


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        sampling.uniform_frame_indices(0, 4)
    with pytest.raises(ValueError):
        sampling.segment_indices(10, 0)
