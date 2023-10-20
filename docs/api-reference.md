# API 参考

顶层 `avalign` 只导出纯 NumPy 的核心 API。

## 音频

| 函数 | 说明 |
| --- | --- |
| `mel_spectrogram(signal, ...)` | 梅尔谱 `(n_mels, T)` |
| `log_mel_spectrogram(signal, ...)` | 对数梅尔谱 |
| `mfcc(signal, n_mfcc=20, ...)` | MFCC `(n_mfcc, T)` |

## 对齐

| 函数 / 类 | 说明 |
| --- | --- |
| `cosine_similarity_matrix(a, b)` | 余弦相似度矩阵 `(m, n)` |
| `l2_normalize(x, axis=-1)` | L2 归一化 |
| `info_nce(query, key, temperature=0.07)` | 单向 InfoNCE 损失 |
| `clip_loss(audio, video, temperature=0.07)` | CLIP 对称损失 |
| `nt_xent(audio, video, temperature=0.5)` | NT-Xent 损失 |
| `LinearProjection(in_dim, out_dim, ...)` | NumPy 线性投影头 |
| `AVAlignPipeline(audio_dim, video_dim, ...)` | 端到端对齐流程 |

## 同步

| 函数 | 说明 |
| --- | --- |
| `estimate_offset(x, y, max_lag=None)` | 估计对齐所需滞后 |

## 检索

| 函数 | 说明 |
| --- | --- |
| `search_topk(q, gallery, k=5, metric="cosine")` | Top-K 检索，返回 `(indices, scores)` |
| `retrieve(q, gallery, k=5)` | 仅返回 Top-K 索引 |
| `bidirectional_retrieval(audio, video, k=5)` | 双向检索 |
| `recall_at_k(sim, k, ground_truth=None)` | Recall@K |
| `retrieval_report(sim, ks=(1,5,10))` | 指标汇总字典 |

## 数据

| 函数 | 说明 |
| --- | --- |
| `make_synthetic_embeddings(n, dim=32, ...)` | 合成配对嵌入 |
| `make_synced_envelopes(length, offset, ...)` | 带已知偏移的时间包络对 |

## 可选（PyTorch，`avalign.models` / `avalign.train`）

| 名称 | 说明 |
| --- | --- |
| `models.AudioEncoder` | 对数梅尔谱上的 2D CNN |
| `models.VisualEncoder` | 帧序列上的 2D CNN + 时间平均 |
| `models.ProjectionHead` | CLIP 风格投影头 |
| `models.AVAlignModel` | 双编码器 + 可学习温度 |
| `train.losses.clip_contrastive_loss` | torch 版对称对比损失 |
| `train.loop.fit` | 最小训练循环 |
