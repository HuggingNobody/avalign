import dataclasses

import pytest

from avalign.config import AlignConfig, AudioConfig, AVConfig


def test_defaults():
    cfg = AVConfig()
    assert cfg.audio.sample_rate == 16000
    assert cfg.align.temperature > 0
    assert cfg.retrieval.topk == (1, 5, 10)


def test_audio_validation():
    with pytest.raises(ValueError):
        AudioConfig(sample_rate=0)
    with pytest.raises(ValueError):
        AudioConfig(n_fft=0)


def test_align_validation():
    with pytest.raises(ValueError):
        AlignConfig(temperature=0.0)
    with pytest.raises(ValueError):
        AlignConfig(loss="bogus")


def test_configs_are_frozen():
    cfg = AudioConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.sample_rate = 8000  # type: ignore[misc]
