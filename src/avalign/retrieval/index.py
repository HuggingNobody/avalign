"""暴力检索索引：把嵌入存起来，按余弦/点积做 Top-K 搜索。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from avalign.align.similarity import cosine_similarity_matrix


class EmbeddingIndex:
    """一个简单的内存向量索引。

    刻意保持朴素（暴力打分）——它是评估检索指标的参考实现，而不是要跟
    FAISS 之类的近似最近邻库拼吞吐。
    """

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in {"cosine", "dot"}:
            raise ValueError(f"未知的度量: {metric}")
        self.metric = metric
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: Sequence[str] | None = None) -> EmbeddingIndex:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)
        if ids is None:
            start = len(self._ids)
            ids = [str(start + i) for i in range(len(vectors))]
        self._ids.extend(ids)
        return self

    def __len__(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def similarity(self, queries: np.ndarray) -> np.ndarray:
        if self._vectors is None:
            raise ValueError("索引为空")
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        if self.metric == "cosine":
            return cosine_similarity_matrix(queries, self._vectors)
        return queries @ self._vectors.T

    def search(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(indices, scores)``，形状均为 ``(num_queries, k)``。"""
        sims = self.similarity(queries)
        k = min(k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return idx, scores
