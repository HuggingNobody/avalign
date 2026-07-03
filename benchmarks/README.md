# 基准

一些粗略的性能参考脚本，用来了解各环节的量级，不进 CI。

```bash
python benchmarks/bench_retrieval.py
```

`bench_retrieval.py` 测量暴力 Top-K 检索在不同 gallery 规模下的耗时与吞吐。
注意索引是刻意保持朴素的暴力实现，规模很大时应换成近似最近邻后端。
