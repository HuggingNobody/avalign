"""框架的全局配置对象。

所有配置都是不可变的 dataclass，方便在实验之间复制与哈希。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    """音频前端参数。"""

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 64
    n_mfcc: int = 20
    fmin: float = 0.0
    fmax: float | None = None
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate 必须为正数")
        if self.hop_length <= 0 or self.n_fft <= 0:
            raise ValueError("n_fft 与 hop_length 必须为正数")


@dataclass(frozen=True)
class VideoConfig:
    """视频前端参数。"""

    fps: float = 25.0
    frame_size: int = 112
    num_frames: int = 16


@dataclass(frozen=True)
class AlignConfig:
    """对比对齐参数。"""

    embed_dim: int = 256
    temperature: float = 0.07
    loss: str = "info_nce"

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature 必须为正数")
        if self.loss not in {"info_nce", "clip", "nt_xent"}:
            raise ValueError(f"未知的损失类型: {self.loss}")


@dataclass(frozen=True)
class RetrievalConfig:
    """跨模态检索评估参数。"""

    metric: str = "cosine"
    topk: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True)
class AVConfig:
    """把各子模块的配置聚合在一起。"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
