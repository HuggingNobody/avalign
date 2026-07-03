import numpy as np

from avalign.retrieval import metrics


def test_rank_perfect_alignment():
    sim = np.eye(5)
    ranks = metrics.rank_of_matches(sim)
    assert np.array_equal(ranks, np.ones(5, dtype=int))


def test_recall_perfect():
    sim = np.eye(8)
    assert metrics.recall_at_k(sim, k=1) == 1.0
    assert metrics.mean_average_precision(sim) == 1.0


def test_median_and_mean_rank():
    # 反对角相似度：正确项都在最后一名
    sim = np.fliplr(np.eye(4))
    gt = np.arange(4)
    ranks = metrics.rank_of_matches(sim, gt)
    assert metrics.median_rank(sim, gt) == float(np.median(ranks))


def test_report_keys():
    sim = np.random.default_rng(0).standard_normal((6, 6))
    report = metrics.retrieval_report(sim, ks=(1, 5))
    assert set(report) == {"R@1", "R@5", "median_rank", "mean_rank", "mAP"}
    assert 0.0 <= report["R@1"] <= 1.0
