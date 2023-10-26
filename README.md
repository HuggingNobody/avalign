# avalign

[![CI](https://github.com/HuggingNobody/avalign/actions/workflows/ci.yml/badge.svg)](https://github.com/HuggingNobody/avalign/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**音视频跨模态对齐与检索框架**——用对比学习把音频编码器与视频/图像编码器对齐到同一个
嵌入空间，并在其上做音视频同步与跨模态检索。

核心算法以纯 NumPy 实现，`import avalign` 不依赖任何深度学习框架；需要可训练的
神经网络编码器时，再按需安装 PyTorch 可选后端。这样既方便在没有 GPU 的机器上做
原型与教学，也保留了走向真实训练的通道。

## 特性

- **音频前端**：分帧、加窗、STFT、梅尔/对数梅尔谱、MFCC、SpecAugment。
- **视觉前端**：帧采样（均匀 / TSN 分段）、灰度、颜色直方图、运动能量、网格池化。
- **对比对齐**：InfoNCE、CLIP 对称损失、NT-Xent，配套 softmax/交叉熵与解析梯度。
- **音视频同步**：基于归一化互相关的偏移估计与容差评估。
- **跨模态检索**：暴力向量索引、双向 Top-K 检索，Recall@K / 中位数排名 / mAP。
- **可选 PyTorch 后端**：音频 CNN、视觉 CNN、CLIP 风格投影头与双编码器训练循环。

## 安装

```bash
pip install avalign            # 仅核心（NumPy）
pip install "avalign[torch]"   # 附带可训练的 PyTorch 编码器
```

## 快速开始

```python
import numpy as np
from avalign import (
    make_synthetic_embeddings,
    clip_loss,
    cosine_similarity_matrix,
    retrieval_report,
)

# 生成一批共享隐空间的合成音视频对
batch = make_synthetic_embeddings(200, dim=64, noise=0.08, seed=0)

# 对齐信号：对角线上的配对样本损失更低
loss = clip_loss(batch.audio_embed, batch.video_embed, temperature=0.07)

# 跨模态检索评估
sim = cosine_similarity_matrix(batch.audio_embed, batch.video_embed)
print(retrieval_report(sim, ks=(1, 5, 10)))
```

或者直接跑端到端演示：

```bash
avalign demo
```

更多用法见 [docs/usage.md](docs/usage.md)。

## 定位

这是一个偏研究向的框架，围绕「音频 ↔ 视觉」的对齐、同步与检索三条线索组织，
方便从硕士阶段的小实验逐步扩展到更完整的博士方向课题。
