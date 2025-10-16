from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn

from .frame_encoder import FrameEncoder
from .temporal_encoder import TemporalEncoder


class Combined(nn.Module):
    """Sequentially apply spatial and temporal encoders.

    The combined model first processes frames with :class:`FrameEncoder` to
    obtain per-frame embeddings and visibility scores, and then feeds the
    resulting sequence into :class:`TemporalEncoder` to predict a heart-rate
    signal.
    """

    def __init__(
        self,
        *,
        frame_encoder: Optional[FrameEncoder] = None,
        temporal_encoder: Optional[TemporalEncoder] = None,
    ) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder or FrameEncoder()
        self.temporal_encoder = temporal_encoder or TemporalEncoder(
            feature_dim=self.frame_encoder.embed_dim
        )

    def forward(self, frames: Tensor, metadata: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Run the spatial encoder followed by the temporal encoder.

        Args:
            frames: Either ``[B, S, C, H, W]`` or ``[S, C, H, W]`` tensors of
                RGB frames with ``C == 3``.
            metadata: Either ``[B, S, 5]`` or ``[S, 5]`` tensors containing the
                per-frame face metadata aligned with ``frames``.

        Returns:
            A tuple ``(predictions, features, visibility)`` where ``predictions``
            contains the heart-rate signal estimated by the temporal encoder,
            ``features`` contains the per-frame embeddings produced by the frame
            encoder, and ``visibility`` provides the visibility weights used for
            temporal attention.
        """

        squeeze_batch = frames.dim() == 4
        if squeeze_batch:
            frames = frames.unsqueeze(0)
            metadata = metadata.unsqueeze(0)

        features, visibility = self.frame_encoder(frames, metadata)
        if features.dim() == 2:
            features = features.unsqueeze(0)
            visibility = visibility.unsqueeze(0)

        predictions = self.temporal_encoder(features, visibility)

        if squeeze_batch:
            predictions = predictions.squeeze(0)
            features = features.squeeze(0)
            visibility = visibility.squeeze(0)

        return predictions, features, visibility
