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


def test_perfect_baseline(synthetic_pairs):
    from avalign.align.similarity import cosine_similarity_matrix

    audio, video = synthetic_pairs
    sim = cosine_similarity_matrix(audio, audio)  # 自检索：每个 query 就是自己
    assert metrics.recall_at_k(sim, k=1) == 1.0
    assert metrics.mean_rank(sim) == 1.0


def test_random_baseline_is_near_chance():
    # 完全随机的相似度：R@1 应接近 1/N
    rng = np.random.default_rng(0)
    n = 200
    recalls = []
    for _ in range(5):
        sim = rng.standard_normal((n, n))
        recalls.append(metrics.recall_at_k(sim, k=1))
    assert np.mean(recalls) < 0.05  # 远低于对齐时的水平


def test_recall_monotonic_in_k(synthetic_pairs):
    from avalign.align.similarity import cosine_similarity_matrix

    audio, video = synthetic_pairs
    sim = cosine_similarity_matrix(audio, video)
    r1 = metrics.recall_at_k(sim, 1)
    r5 = metrics.recall_at_k(sim, 5)
    r10 = metrics.recall_at_k(sim, 10)
    assert r1 <= r5 <= r10
