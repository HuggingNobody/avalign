import pytest

torch = pytest.importorskip("torch")

from avalign.models import AudioEncoder, AVAlignModel, ProjectionHead, VisualEncoder


def test_audio_encoder_output_shape():
    enc = AudioEncoder(n_mels=64, embed_dim=128)
    x = torch.randn(4, 64, 80)
    assert enc(x).shape == (4, 128)


def test_visual_encoder_single_frame():
    enc = VisualEncoder(embed_dim=64)
    x = torch.randn(3, 3, 32, 32)
    assert enc(x).shape == (3, 64)


def test_visual_encoder_frame_sequence():
    enc = VisualEncoder(embed_dim=64)
    x = torch.randn(2, 5, 3, 32, 32)
    assert enc(x).shape == (2, 64)


def test_projection_head_shape():
    head = ProjectionHead(dim=100, proj_dim=64)
    assert head(torch.randn(8, 100)).shape == (8, 64)


def test_av_model_logits_are_square():
    model = AVAlignModel(embed_dim=32)
    audio = torch.randn(6, 64, 60)
    video = torch.randn(6, 3, 32, 32)
    logits = model(audio, video)
    assert logits.shape == (6, 6)
