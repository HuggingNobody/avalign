"""跨模态检索评估指标：Recall@K、中位数排名、平均排名与 mAP。

约定每个 query 只有一个正确项（音视频一一配对），因此平均精度退化为
``1 / rank``，mAP 即所有 query 的 ``mean(1 / rank)``。
"""

from __future__ import annotations

import numpy as np


def rank_of_matches(sim: np.ndarray, ground_truth: np.ndarray | None = None) -> np.ndarray:
    """每个 query 的正确项排名（1 起始）。``sim`` 形状 ``(num_queries, gallery)``。"""
    sim = np.atleast_2d(np.asarray(sim, dtype=float))
    num_queries = sim.shape[0]
    if ground_truth is None:
        ground_truth = np.arange(num_queries)
    ground_truth = np.asarray(ground_truth)

    order = np.argsort(-sim, axis=1)
    ranks = np.empty(num_queries, dtype=int)
    for i in range(num_queries):
        ranks[i] = int(np.where(order[i] == ground_truth[i])[0][0]) + 1
    return ranks


def recall_at_k(sim: np.ndarray, k: int, ground_truth: np.ndarray | None = None) -> float:
    """Recall@K：正确项落在前 K 名的 query 比例。"""
    ranks = rank_of_matches(sim, ground_truth)
    return float(np.mean(ranks <= k))


def median_rank(sim: np.ndarray, ground_truth: np.ndarray | None = None) -> float:
    ranks = rank_of_matches(sim, ground_truth)
    return float(np.median(ranks))


def mean_rank(sim: np.ndarray, ground_truth: np.ndarray | None = None) -> float:
    ranks = rank_of_matches(sim, ground_truth)
    return float(np.mean(ranks))


def mean_average_precision(sim: np.ndarray, ground_truth: np.ndarray | None = None) -> float:
    ranks = rank_of_matches(sim, ground_truth)
    return float(np.mean(1.0 / ranks))


def retrieval_report(
    sim: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    ground_truth: np.ndarray | None = None,
) -> dict[str, float]:
    """把常用指标汇总成一个字典，方便打印或记录。"""
    ranks = rank_of_matches(sim, ground_truth)
    report: dict[str, float] = {f"R@{k}": float(np.mean(ranks <= k)) for k in ks}
    report["median_rank"] = float(np.median(ranks))
    report["mean_rank"] = float(np.mean(ranks))
    report["mAP"] = float(np.mean(1.0 / ranks))
    return report
