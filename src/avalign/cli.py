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
from avalign.data.synthetic import make_synced_envelopes, make_synthetic_embeddings
from avalign.retrieval.metrics import retrieval_report
from avalign.sync.offset import estimate_offset


def _cmd_features(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    signal = rng.standard_normal(args.samples)
    log_mel = log_mel_spectrogram(
        signal, sample_rate=args.sample_rate, n_mels=args.n_mels
    )
    print(f"对数梅尔谱形状: {log_mel.shape}")
    print(f"均值={log_mel.mean():.4f} 标准差={log_mel.std():.4f}")
    return 0


def _cmd_retrieve(args: argparse.Namespace) -> int:
    batch = make_synthetic_embeddings(
        args.num_pairs, dim=args.dim, noise=args.noise, seed=args.seed
    )
    from avalign.align.similarity import cosine_similarity_matrix

    sim = cosine_similarity_matrix(batch.audio_embed, batch.video_embed)
    report = retrieval_report(sim, ks=(1, 5, 10))
    print(f"合成 {args.num_pairs} 对样本，噪声={args.noise}")
    for key, value in report.items():
        print(f"  {key}: {value:.4f}")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    audio, video, true_offset = make_synced_envelopes(
        length=args.length, offset=args.offset, noise=args.noise, seed=args.seed
    )
    est = estimate_offset(video, audio, max_lag=args.max_lag)
    print(f"真实偏移={true_offset}  估计偏移={est}  误差={abs(est - true_offset)}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    print("=== avalign 端到端演示 ===")
    batch = make_synthetic_embeddings(args.num_pairs, dim=32, noise=0.08, seed=args.seed)
    from avalign.align.similarity import cosine_similarity_matrix

    sim = cosine_similarity_matrix(batch.audio_embed, batch.video_embed)
    report = retrieval_report(sim, ks=(1, 5, 10))
    print("[检索] 合成音视频对的跨模态检索指标：")
    for key, value in report.items():
        print(f"  {key}: {value:.4f}")

    audio, video, true_offset = make_synced_envelopes(length=300, offset=9, seed=args.seed)
    est = estimate_offset(video, audio, max_lag=50)
    print(f"[同步] 真实偏移={true_offset}  估计偏移={est}")
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

    p_ret = sub.add_parser("retrieve", help="在合成音视频对上评估跨模态检索")
    p_ret.add_argument("--num-pairs", type=int, default=100)
    p_ret.add_argument("--dim", type=int, default=64)
    p_ret.add_argument("--noise", type=float, default=0.1)
    p_ret.add_argument("--seed", type=int, default=0)
    p_ret.set_defaults(func=_cmd_retrieve)

    p_sync = sub.add_parser("sync", help="估计合成音视频包络之间的时间偏移")
    p_sync.add_argument("--length", type=int, default=300)
    p_sync.add_argument("--offset", type=int, default=7)
    p_sync.add_argument("--noise", type=float, default=0.05)
    p_sync.add_argument("--max-lag", type=int, default=50)
    p_sync.add_argument("--seed", type=int, default=0)
    p_sync.set_defaults(func=_cmd_sync)

    p_demo = sub.add_parser("demo", help="一次性跑通检索与同步的端到端演示")
    p_demo.add_argument("--num-pairs", type=int, default=100)
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.set_defaults(func=_cmd_demo)

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
