"""信号分帧与短时傅里叶变换。

TODO: 加窗函数、STFT 与功率谱。
"""

from __future__ import annotations

import numpy as np


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """把一维信号切成互相重叠的帧。"""
    raise NotImplementedError
