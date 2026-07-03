"""avalign: 音视频跨模态对齐与检索框架。

顶层只导出纯 NumPy 的核心 API，``import avalign`` 不会拉起 PyTorch。
需要可训练编码器时，从 :mod:`avalign.models` 与 :mod:`avalign.train`
（可选依赖 ``torch``）显式导入。
"""

from __future__ import annotations

from avalign.align.contrastive import clip_loss, info_nce, nt_xent
from avalign.align.projection import LinearProjection
from avalign.align.similarity import cosine_similarity_matrix, l2_normalize
from avalign.audio.features import log_mel_spectrogram, mel_spectrogram, mfcc
from avalign.config import (
    AlignConfig,
    AudioConfig,
    AVConfig,
    RetrievalConfig,
    VideoConfig,
)
from avalign.data.synthetic import make_synced_envelopes, make_synthetic_embeddings
from avalign.pipeline import AVAlignPipeline
from avalign.retrieval.metrics import recall_at_k, retrieval_report
from avalign.retrieval.search import bidirectional_retrieval, retrieve, search_topk
from avalign.sync.offset import estimate_offset

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "AVConfig",
    "AudioConfig",
    "VideoConfig",
    "AlignConfig",
    "RetrievalConfig",
    "mel_spectrogram",
    "log_mel_spectrogram",
    "mfcc",
    "cosine_similarity_matrix",
    "l2_normalize",
    "info_nce",
    "clip_loss",
    "nt_xent",
    "LinearProjection",
    "AVAlignPipeline",
    "search_topk",
    "retrieve",
    "bidirectional_retrieval",
    "recall_at_k",
    "retrieval_report",
    "estimate_offset",
    "make_synthetic_embeddings",
    "make_synced_envelopes",
]
