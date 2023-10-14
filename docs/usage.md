# 使用指南

## 抽取音频特征

```python
import numpy as np
from avalign import log_mel_spectrogram, mfcc

signal = np.random.randn(16000)               # 1 秒、16kHz
log_mel = log_mel_spectrogram(signal, sample_rate=16000, n_mels=64)
coeffs = mfcc(signal, n_mfcc=20, n_mels=64)
print(log_mel.shape, coeffs.shape)            # (64, T) (20, T)
```

## 对齐一批音视频对

```python
from avalign import make_synthetic_embeddings, clip_loss, cosine_similarity_matrix

batch = make_synthetic_embeddings(64, dim=32, noise=0.08, seed=0)
loss = clip_loss(batch.audio_embed, batch.video_embed, temperature=0.07)
sim = cosine_similarity_matrix(batch.audio_embed, batch.video_embed)
```

## 跨模态检索评估

```python
from avalign import retrieval_report

report = retrieval_report(sim, ks=(1, 5, 10))
print(report)   # {'R@1': ..., 'R@5': ..., 'median_rank': ..., 'mAP': ...}
```

## 估计音视频同步偏移

```python
from avalign import make_synced_envelopes, estimate_offset

audio_env, video_env, true_offset = make_synced_envelopes(length=300, offset=7)
est = estimate_offset(video_env, audio_env, max_lag=50)
```

## 命令行

所有子命令都在合成数据上自成一体，无需外部音视频文件：

```bash
avalign features --n-mels 64      # 抽取对数梅尔谱
avalign retrieve --num-pairs 200  # 跨模态检索指标
avalign sync --offset 9           # 同步偏移估计
avalign demo                      # 端到端演示
```

## 使用 PyTorch 后端

```python
from avalign.models import AVAlignModel
from avalign.train.loop import fit
import torch

model = AVAlignModel(embed_dim=256)
audio = torch.randn(8, 64, 100)      # (B, n_mels, T)
video = torch.randn(8, 3, 112, 112)  # (B, C, H, W)
history = fit(model, [(audio, video)], epochs=5)
```
