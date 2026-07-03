import pytest

torch = pytest.importorskip("torch")

from avalign.models import AVAlignModel
from avalign.train.loop import fit
from avalign.train.losses import clip_contrastive_loss, info_nce_loss


def test_clip_contrastive_loss_low_on_perfect_logits():
    logits = torch.eye(5) * 10.0
    loss = clip_contrastive_loss(logits)
    assert loss.item() < 0.1


def test_info_nce_loss_finite():
    a = torch.randn(8, 16)
    v = torch.randn(8, 16)
    loss = info_nce_loss(a, v, temperature=0.07)
    assert torch.isfinite(loss)


def test_fit_returns_history_and_decreases():
    torch.manual_seed(0)
    model = AVAlignModel(embed_dim=32)
    audio = torch.randn(8, 64, 60)
    video = torch.randn(8, 3, 32, 32)
    history = fit(model, [(audio, video)], epochs=12, lr=1e-3)
    assert len(history) == 12
    # 在同一个 batch 上反复训练应能过拟合，损失下降
    assert history[-1] < history[0]
