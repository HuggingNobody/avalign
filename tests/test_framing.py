import numpy as np

from avalign.audio import framing


def test_hann_window_endpoints():
    w = framing.hann_window(8)
    assert w.shape == (8,)
    assert w[0] == 0.0
    assert np.isclose(w.max(), 1.0, atol=1e-6)


def test_frame_signal_shape():
    sig = np.arange(20, dtype=float)
    frames = framing.frame_signal(sig, frame_length=8, hop_length=4)
    assert frames.shape == (4, 8)
    # 第二帧应从索引 4 开始
    assert frames[1, 0] == 4.0


def test_frame_signal_pads_short_input():
    sig = np.ones(3)
    frames = framing.frame_signal(sig, frame_length=8, hop_length=4)
    assert frames.shape == (1, 8)


def test_stft_shape():
    sig = np.random.default_rng(0).standard_normal(2000)
    spec = framing.stft(sig, n_fft=400, hop_length=160)
    assert spec.shape[0] == 400 // 2 + 1
    assert np.iscomplexobj(spec)


def test_power_spectrum_nonnegative():
    sig = np.random.default_rng(1).standard_normal(2000)
    ps = framing.power_spectrum(sig, n_fft=256, hop_length=128)
    assert np.all(ps >= 0.0)


def test_unknown_window_raises():
    import pytest

    with pytest.raises(ValueError):
        framing.get_window("blackman", 16)
