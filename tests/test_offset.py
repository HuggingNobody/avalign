import numpy as np

from avalign.sync import offset


def _delayed(signal, delay):
    """把 signal 延迟 delay 个采样点（前面补零）。"""
    out = np.zeros(len(signal) + delay)
    out[delay : delay + len(signal)] = signal
    return out


def test_estimate_positive_offset():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(200)
    y = np.concatenate([base, np.zeros(5)])
    x = _delayed(base, 5)
    assert offset.estimate_offset(x, y, max_lag=20) == 5


def test_zero_offset_for_identical():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(128)
    assert offset.estimate_offset(sig, sig.copy(), max_lag=10) == 0


def test_align_by_offset_overlaps():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(100)
    x = _delayed(base, 8)
    y = np.concatenate([base, np.zeros(8)])
    est = offset.estimate_offset(x, y, max_lag=20)
    xa, ya = offset.align_by_offset(x, y, est)
    assert len(xa) == len(ya)
    assert np.corrcoef(xa, ya)[0, 1] > 0.99


def test_cross_correlation_shapes():
    lags, corr = offset.cross_correlation(np.arange(10.0), np.arange(10.0), max_lag=4)
    assert lags.shape == corr.shape == (9,)


def test_negative_offset_recovery():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(200)
    # x 领先 y 6 个采样点 → 需要的滞后为 -6
    x = np.concatenate([base, np.zeros(6)])
    y = _delayed(base, 6)
    assert offset.estimate_offset(x, y, max_lag=20) == -6


def test_offset_bounded_by_max_lag():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    est = offset.estimate_offset(x, y, max_lag=5)
    assert -5 <= est <= 5
