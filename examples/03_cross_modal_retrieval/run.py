"""示例 03：合成音视频对的跨模态检索评估。"""

from avalign import (
    cosine_similarity_matrix,
    make_synthetic_embeddings,
    retrieval_report,
)


def main() -> None:
    for noise in (0.05, 0.2, 0.5):
        batch = make_synthetic_embeddings(200, dim=64, noise=noise, seed=0)
        sim = cosine_similarity_matrix(batch.audio_embed, batch.video_embed)
        report = retrieval_report(sim, ks=(1, 5, 10))
        line = "  ".join(f"{k}={v:.3f}" for k, v in report.items())
        print(f"noise={noise:>4}:  {line}")


if __name__ == "__main__":
    main()
