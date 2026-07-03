"""跨模态检索吞吐基准。

对不同 gallery 规模测量暴力 Top-K 检索的耗时。这是一个粗略的量级参考，
而非严谨的 profiling。
"""

import time

import numpy as np

from avalign.retrieval.search import search_topk


def bench(num_gallery: int, num_query: int, dim: int = 256, repeats: int = 3) -> None:
    rng = np.random.default_rng(0)
    gallery = rng.standard_normal((num_gallery, dim))
    query = rng.standard_normal((num_query, dim))

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        search_topk(query, gallery, k=10)
        best = min(best, time.perf_counter() - start)

    qps = num_query / best
    print(
        f"gallery={num_gallery:>6}  query={num_query:>5}  dim={dim}: "
        f"{best * 1e3:8.2f} ms  ({qps:,.0f} q/s)"
    )


def main() -> None:
    for num_gallery in (1_000, 5_000, 20_000):
        bench(num_gallery, num_query=1_000)


if __name__ == "__main__":
    main()
