# 架构

avalign 按「模态前端 → 对齐 → 下游任务」的思路分层，核心层不依赖任何深度学习框架。

```
avalign/
├── config.py           # 不可变配置对象（音频/视频/对齐/检索）
├── audio/              # 音频前端：framing / features / augment
├── video/              # 视觉前端：sampling / features / transforms
├── align/              # 对齐核心：similarity / contrastive / projection
├── sync/               # 音视频同步：offset / metrics
├── retrieval/          # 跨模态检索：store / search / metrics
├── data/               # 数据容器与合成数据：pairs / synthetic
├── pipeline.py         # 端到端对齐流程（纯 NumPy）
├── cli.py              # 命令行入口
├── models/             # 可选：PyTorch 编码器（惰性加载）
└── train/              # 可选：PyTorch 损失与训练循环
```

## 分层原则

1. **数学在 NumPy 层。** 所有特征、损失、指标、同步的数学都用 NumPy 写，
   保证 `import avalign` 不牵连 torch，约八成测试无需深度学习依赖即可运行。
2. **神经网络在可选层。** `models/` 与 `train/` 依赖 `torch`，通过 `models/__init__.py`
   的模块级 `__getattr__` 惰性加载，只有显式访问才导入。
3. **配置即数据。** 配置都是 frozen dataclass，可哈希、可复制，便于在实验之间传递。

## 数据流

音频波形 →（framing / STFT / 梅尔谱）→ 音频特征
帧序列  →（采样 / 颜色·运动特征）→ 视觉特征
两路特征 → 投影 + L2 归一化 → 共享空间嵌入
共享嵌入 →（相似度矩阵）→ 对比损失 / 检索指标
时间包络 →（互相关）→ 同步偏移

## 为什么用共享空间

把两个模态映射到同一个单位球面后，余弦相似度既是训练时的对齐信号，也是
推理时的检索打分，同一套表示同时服务于对齐、检索与（配合时间包络的）同步。
