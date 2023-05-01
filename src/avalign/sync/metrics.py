"""同步评估：偏移误差与容差内准确率。"""

from __future__ import annotations

import numpy as np


def offset_error(pred_offset: float, true_offset: float) -> float:
    """单个样本的绝对偏移误差。"""
    return abs(float(pred_offset) - float(true_offset))


def within_tolerance(pred_offset: float, true_offset: float, tol: float) -> bool:
    """预测偏移是否落在容差范围内。"""
    return offset_error(pred_offset, true_offset) <= tol


def sync_accuracy(preds: np.ndarray, trues: np.ndarray, tol: float) -> float:
    """一批样本在给定容差下的同步准确率。"""
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    if preds.shape != trues.shape:
        raise ValueError("预测与真值的形状必须一致")
    return float(np.mean(np.abs(preds - trues) <= tol))


def frames_to_seconds(num_frames: float, fps: float) -> float:
    """把以帧为单位的偏移换算成秒。"""
    if fps <= 0:
        raise ValueError("fps 必须为正数")
    return float(num_frames) / fps
