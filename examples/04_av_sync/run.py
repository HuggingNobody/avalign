"""示例 04：从时间包络估计音视频同步偏移。"""

from avalign import estimate_offset, make_synced_envelopes


def main() -> None:
    for true_offset in (-8, 0, 5, 12):
        audio, video, _ = make_synced_envelopes(
            length=400, offset=true_offset, noise=0.05, seed=0
        )
        est = estimate_offset(video, audio, max_lag=40)
        flag = "ok" if abs(est - true_offset) <= 1 else "miss"
        print(f"真实偏移={true_offset:>4}  估计={est:>4}  [{flag}]")


if __name__ == "__main__":
    main()
