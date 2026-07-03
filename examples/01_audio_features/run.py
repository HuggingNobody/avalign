"""示例 01：从一段合成音频抽取对数梅尔谱与 MFCC。"""

import numpy as np

from avalign import log_mel_spectrogram, mfcc


def main() -> None:
    # 合成一个 1 秒、220Hz 的正弦音，采样率 16kHz
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = np.sin(2 * np.pi * 220 * t) + 0.01 * np.random.default_rng(0).standard_normal(sr)

    log_mel = log_mel_spectrogram(signal, sample_rate=sr, n_mels=64)
    coeffs = mfcc(signal, n_mfcc=20, sample_rate=sr, n_mels=64)

    print(f"对数梅尔谱: {log_mel.shape}  (n_mels, frames)")
    print(f"MFCC:       {coeffs.shape}  (n_mfcc, frames)")
    print(f"对数梅尔谱能量最高的频带索引: {log_mel.mean(axis=1).argmax()}")


if __name__ == "__main__":
    main()
