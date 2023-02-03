import numpy as np
import pytest

from avalign.audio import features


def test_mel_filterbank_shape():
    fb = features.mel_filterbank(n_mels=40, n_fft=400, sample_rate=16000)
    assert fb.shape == (40, 201)
    assert np.all(fb >= 0.0)


def test_hz_mel_roundtrip():
    hz = np.array([100.0, 500.0, 2000.0, 8000.0])
    back = features.mel_to_hz(features.hz_to_mel(hz))
    assert np.allclose(back, hz, rtol=1e-4)


def test_mel_spectrogram_shape():
    sig = np.random.default_rng(0).standard_normal(16000)
    mel = features.mel_spectrogram(sig, n_mels=64, n_fft=400, hop_length=160)
    assert mel.shape[0] == 64
    assert np.all(mel >= 0.0)


def test_log_mel_is_finite():
    sig = np.zeros(4000)
    log_mel = features.log_mel_spectrogram(sig, n_mels=32)
    assert np.all(np.isfinite(log_mel))


def test_mfcc_shape():
    sig = np.random.default_rng(1).standard_normal(8000)
    coeffs = features.mfcc(sig, n_mfcc=13, n_mels=40)
    assert coeffs.shape[0] == 13


def test_mfcc_rejects_too_many_coeffs():
    sig = np.random.default_rng(2).standard_normal(4000)
    with pytest.raises(ValueError):
        features.mfcc(sig, n_mfcc=50, n_mels=40)
