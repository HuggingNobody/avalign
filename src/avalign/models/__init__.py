"""可训练的 PyTorch 编码器（可选依赖 ``torch``）。

这些模块通过模块级 ``__getattr__`` 惰性加载，因此 ``import avalign`` 以及
``import avalign.models`` 本身都不会触发 ``import torch``——只有真正访问某个
类时才会加载对应实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["AudioEncoder", "VisualEncoder", "ProjectionHead", "AVAlignModel"]

if TYPE_CHECKING:
    from avalign.models.audio_encoder import AudioEncoder
    from avalign.models.av_model import AVAlignModel
    from avalign.models.projection import ProjectionHead
    from avalign.models.visual_encoder import VisualEncoder


def __getattr__(name: str) -> Any:
    if name == "AudioEncoder":
        from avalign.models.audio_encoder import AudioEncoder

        return AudioEncoder
    if name == "VisualEncoder":
        from avalign.models.visual_encoder import VisualEncoder

        return VisualEncoder
    if name == "ProjectionHead":
        from avalign.models.projection import ProjectionHead

        return ProjectionHead
    if name == "AVAlignModel":
        from avalign.models.av_model import AVAlignModel

        return AVAlignModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
