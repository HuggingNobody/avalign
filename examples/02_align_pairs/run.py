"""示例 02：在合成音视频对上比较对齐前后的对比损失。"""

import numpy as np

from avalign import clip_loss, cosine_similarity_matrix, make_synthetic_embeddings


def main() -> None:
    batch = make_synthetic_embeddings(64, dim=32, noise=0.08, seed=0)
    audio, video = batch.audio_embed, batch.video_embed

    aligned = clip_loss(audio, video, temperature=0.07)
    shuffled = clip_loss(audio, np.random.default_rng(1).permutation(video, axis=0), 0.07)

    sim = cosine_similarity_matrix(audio, video)
    diag = np.diag(sim).mean()
    off = (sim.sum() - np.trace(sim)) / (sim.size - len(sim))

    print(f"配对 CLIP 损失:   {aligned:.4f}")
    print(f"打乱后 CLIP 损失: {shuffled:.4f}")
    print(f"对角线平均相似度: {diag:.4f}  非对角平均: {off:.4f}")


if __name__ == "__main__":
    main()
