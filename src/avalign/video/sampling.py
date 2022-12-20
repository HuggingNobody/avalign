"""从视频帧序列中采样关键帧。

TODO: 均匀采样与 TSN 式分段采样。
"""

from __future__ import annotations

import numpy as np


def uniform_frame_indices(num_available: int, num_sampled: int) -> np.ndarray:
    """在 [0, num_available) 上均匀取 num_sampled 个索引。"""
    raise NotImplementedError
