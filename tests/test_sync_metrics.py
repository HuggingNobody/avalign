import numpy as np
import pytest

from avalign.sync import metrics


def test_offset_error():
    assert metrics.offset_error(5, 3) == 2.0


def test_within_tolerance():
    assert metrics.within_tolerance(5, 6, tol=2)
    assert not metrics.within_tolerance(5, 10, tol=2)


def test_sync_accuracy():
    preds = np.array([0, 1, 5, 3])
    trues = np.array([0, 2, 5, 9])
    # 误差 [0,1,0,6]，容差 1 → 命中 3/4
    assert metrics.sync_accuracy(preds, trues, tol=1) == 0.75


def test_sync_accuracy_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.sync_accuracy(np.zeros(3), np.zeros(4), tol=1)


def test_frames_to_seconds():
    assert metrics.frames_to_seconds(50, fps=25.0) == 2.0
    with pytest.raises(ValueError):
        metrics.frames_to_seconds(10, fps=0)
