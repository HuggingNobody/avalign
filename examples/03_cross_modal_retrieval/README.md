# 示例 03：跨模态检索

在不同噪声水平下生成合成音视频对，用暴力索引做音频→视频检索，并打印
Recall@K、中位数排名与 mAP。

```bash
python examples/03_cross_modal_retrieval/run.py
```

随着噪声增大，配对信号被淹没，各项指标应随之下降——这正好说明指标是有区分度的。
