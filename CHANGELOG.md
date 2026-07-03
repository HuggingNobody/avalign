# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 风格，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-07-03

### Added
- 音频前端：分帧、加窗、STFT、梅尔/对数梅尔谱、MFCC、SpecAugment。
- 视觉前端：均匀 / TSN 分段采样、灰度、颜色直方图、运动能量、网格池化。
- 对齐核心：余弦相似度、InfoNCE、CLIP 对称损失、NT-Xent、NumPy 投影头。
- 音视频同步：归一化互相关偏移估计与容差评估。
- 跨模态检索：暴力向量索引、双向 Top-K、Recall@K / 中位数排名 / mAP。
- 合成数据生成器与端到端对齐流程 `AVAlignPipeline`。
- `avalign` 命令行：`features` / `retrieve` / `sync` / `demo`。
- 可选 PyTorch 后端：音频/视觉 CNN 编码器、投影头、双编码器与训练循环。

[Unreleased]: https://github.com/HuggingNobody/avalign/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HuggingNobody/avalign/releases/tag/v0.1.0
