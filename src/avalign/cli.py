"""avalign 命令行入口。

各子命令都在合成数据上自成一体地运行，无需外部音视频文件即可体验
特征抽取、检索与同步的完整链路。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from avalign import __version__
from avalign.audio.features import log_mel_spectrogram


def _cmd_features(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    signal = rng.standard_normal(args.samples)
    log_mel = log_mel_spectrogram(
        signal, sample_rate=args.sample_rate, n_mels=args.n_mels
    )
    print(f"对数梅尔谱形状: {log_mel.shape}")
    print(f"均值={log_mel.mean():.4f} 标准差={log_mel.std():.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avalign",
        description="音视频跨模态对齐与检索框架",
    )
    parser.add_argument("--version", action="version", version=f"avalign {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_feat = sub.add_parser("features", help="从合成音频抽取对数梅尔谱")
    p_feat.add_argument("--samples", type=int, default=16000)
    p_feat.add_argument("--sample-rate", type=int, default=16000)
    p_feat.add_argument("--n-mels", type=int, default=64)
    p_feat.add_argument("--seed", type=int, default=0)
    p_feat.set_defaults(func=_cmd_features)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
