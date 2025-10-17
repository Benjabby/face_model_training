"""Unit tests for the rPPG model components."""

import torch

from face_model_training.models import RPPGModel, RPPGSpatialEncoder, RPPGTemporalEncoder


def test_spatial_encoder_emits_expected_shape() -> None:
    encoder = RPPGSpatialEncoder()
    frames = torch.randn(2, 5, 6, 112, 112)
    metadata = torch.randn(2, 5, 5)

    features, visibility = encoder(frames, metadata)

    assert features.shape == (2, 5, encoder.feature_dim)
    assert visibility.shape == (2, 5)


def test_temporal_encoder_produces_scalar_prediction() -> None:
    temporal = RPPGTemporalEncoder(
        input_dim=8,
        hidden_dim=16,
        num_layers=2,
        metadata_dim=4,
        metadata_hidden_dim=8,
    )
    sequence = torch.randn(3, 10, 8)
    visibility = torch.ones(3, 10)
    metadata = torch.randn(3, 10, 4)

    output = temporal(sequence, visibility, metadata)

    assert output.shape == (3,)


def test_rppg_model_end_to_end() -> None:
    model = RPPGModel(return_scalar_heart_rate=False)
    frames = torch.randn(1, 8, 6, 112, 112)
    metadata = torch.ones(1, 8, 5)

    predictions, features, visibility = model(frames, metadata)

    assert predictions.shape == (1, 8)
    assert features.shape[-1] == model.spatial_encoder.feature_dim
    assert visibility.shape == (1, 8)
