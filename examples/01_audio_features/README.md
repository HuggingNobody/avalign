# 示例 01：音频特征抽取

从一段合成正弦音里抽取对数梅尔谱和 MFCC，并打印各自的形状。

```bash
python examples/01_audio_features/run.py
```

预期输出里对数梅尔谱形状为 `(64, frames)`，MFCC 为 `(20, frames)`。
