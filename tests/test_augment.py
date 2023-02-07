import numpy as np

from avalign.audio import augment


def test_add_noise_changes_signal():
    sig = np.sin(np.linspace(0, 20, 4000))
    noisy = augment.add_noise(sig, snr_db=10.0, rng=0)
    assert noisy.shape == sig.shape
    assert not np.allclose(noisy, sig)


def test_add_noise_zero_signal_is_stable():
    sig = np.zeros(100)
    out = augment.add_noise(sig, snr_db=5.0, rng=0)
    assert np.allclose(out, 0.0)


def test_add_noise_snr_is_reasonable():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(20000)
    noisy = augment.add_noise(sig, snr_db=20.0, rng=1)
    noise = noisy - sig
    measured = 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
    assert abs(measured - 20.0) < 1.5


def test_spec_augment_masks_some_values():
    spec = np.ones((40, 80))
    out = augment.spec_augment(
        spec, num_freq_masks=2, freq_mask_width=5, num_time_masks=2, time_mask_width=10, rng=0
    )
    assert out.shape == spec.shape
    assert (out == 0.0).any()


def test_spec_augment_does_not_mutate_input():
    spec = np.ones((10, 10))
    _ = augment.spec_augment(spec, rng=0)
    assert np.all(spec == 1.0)
