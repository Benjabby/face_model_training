"""Spatial-temporal rPPG model components.

This module implements a lightweight remote photoplethysmography (rPPG)
architecture that explicitly separates the spatial and temporal processing
stages.  Classical non-learning rPPG methods first aggregate spatial colour
information from facial regions and subsequently apply temporal filtering to
recover the blood volume pulse.  The :class:`RPPGSpatialEncoder` and
``RPPGTemporalEncoder`` defined here mimic that decomposition while remaining
fully differentiable.

The spatial encoder produces per-frame colour traces by learning an attention
map over the face crop and computing weighted averages of the RGB channels and
their temporal differences.  Additional colour-difference features inspired by
methods such as CHROM are derived to stabilise the signal.  The temporal
encoder then operates purely along the time axis using depthwise-separable
convolutions to model the pulsatile dynamics.  The resulting model can emit
either the complete predicted rPPG waveform or a scalar heart-rate estimate per
window when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _ensure_batch_first(frames: Tensor, metadata: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor], bool]:
    """Convert ``[S, ...]`` tensors to ``[1, S, ...]`` variants."""

    squeeze_batch = frames.dim() == 4
    if squeeze_batch:
        frames = frames.unsqueeze(0)
        if metadata is not None:
            metadata = metadata.unsqueeze(0)
    return frames, metadata, squeeze_batch


class RPPGSpatialEncoder(nn.Module):
    """Encode facial frames into spatially pooled colour traces."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        hidden_channels: int = 32,
        attention_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = attention_kernel_size // 2
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=attention_kernel_size, padding=padding),
        )

        self.register_buffer("_eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False)

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the emitted per-frame features."""

        # RGB mean, RGB differences, and two chrominance ratios.
        return 8

    def forward(self, frames: Tensor, metadata: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Generate spatially pooled features for each frame.

        Args:
            frames: Tensor shaped ``[B, S, C, H, W]`` containing the stacked
                face crops.  The first three channels are expected to represent
                RGB values while the remaining channels hold the per-frame RGB
                differences provided by :mod:`datasets.face_window_dataset`.
            metadata: Optional tensor shaped ``[B, S, 5]`` containing the
                per-frame metadata.  When supplied, the first component is used
                as a visibility indicator for downstream masking.

        Returns:
            A tuple ``(features, visibility)`` where ``features`` has shape
            ``[B, S, feature_dim]`` and contains the pooled colour descriptors
            and ``visibility`` has shape ``[B, S]``.
        """

        frames, metadata, squeeze_batch = _ensure_batch_first(frames, metadata)
        if frames.dim() != 5:
            raise ValueError("frames must have shape [B, S, C, H, W] or [S, C, H, W]")

        batch, seq_len, channels, height, width = frames.shape
        if channels < 3:
            raise ValueError("frames must contain at least RGB channels")

        rgb = frames[..., :3, :, :]
        diffs = frames[..., 3:6, :, :] if channels >= 6 else torch.zeros_like(rgb)

        flat_frames = frames.reshape(batch * seq_len, channels, height, width)
        attention_logits = self.attention_net(flat_frames)
        attention = torch.sigmoid(attention_logits)
        attention = attention.view(batch, seq_len, 1, height, width)

        weighted_rgb = attention * rgb
        weighted_diffs = attention * diffs

        flat_attention = attention.flatten(3)
        rgb_sum = weighted_rgb.flatten(3).sum(dim=-1)
        diff_sum = weighted_diffs.flatten(3).sum(dim=-1)

        eps = self._eps.to(dtype=frames.dtype, device=frames.device)
        normalizer = flat_attention.sum(dim=-1).clamp_min(eps)
        mean_rgb = rgb_sum / normalizer
        mean_diffs = diff_sum / normalizer

        mean_rgb = mean_rgb.to(frames.dtype)
        mean_diffs = mean_diffs.to(frames.dtype)

        # Compute chrominance components similar to the CHROM method for
        # illumination-robust features.
        rgb_norm = mean_rgb / mean_rgb.mean(dim=-1, keepdim=True).clamp_min(eps)
        chrom_x = rgb_norm[..., 0] - rgb_norm[..., 1]
        chrom_y = rgb_norm[..., 0] + rgb_norm[..., 1] - 2 * rgb_norm[..., 2]

        features = torch.cat(
            (
                mean_rgb,
                mean_diffs,
                chrom_x.unsqueeze(-1),
                chrom_y.unsqueeze(-1),
            ),
            dim=-1,
        )

        if metadata is not None:
            if metadata.shape[:2] != (batch, seq_len):
                raise ValueError("metadata must align with frames in the first two dimensions")
            visibility = metadata[..., 0].to(frames.dtype)
        else:
            visibility = torch.ones((batch, seq_len), dtype=frames.dtype, device=frames.device)

        if squeeze_batch:
            features = features.squeeze(0)
            visibility = visibility.squeeze(0)

        return features, visibility


@dataclass
class _TemporalBlockConfig:
    dilation: int
    kernel_size: int


class _TemporalResidualBlock(nn.Module):
    """Depthwise-separable temporal residual block."""

    def __init__(self, channels: int, config: _TemporalBlockConfig, dropout: float) -> None:
        super().__init__()
        padding = (config.kernel_size - 1) // 2 * config.dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.kernel_size,
            dilation=config.dilation,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        residual = x
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        out = residual + y
        if mask is not None:
            out = out * mask
        return out


class RPPGTemporalEncoder(nn.Module):
    """Temporal encoder that models pulse dynamics with 1D convolutions.

    The encoder can optionally incorporate auxiliary per-frame metadata such as
    bounding-box centre coordinates and dimensions.  These values are first
    projected through a small MLP and concatenated to the spatial features
    before the temporal convolutions are applied.  Providing metadata helps the
    model disambiguate motion originating from camera movement or imperfect
    detections from the physiological signal.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 5,
        dilation_growth: int = 2,
        dropout: float = 0.1,
        output_scalar: bool = True,
        metadata_dim: int = 4,
        metadata_hidden_dim: int = 16,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.metadata_dim = metadata_dim
        if metadata_dim > 0:
            self.metadata_proj = nn.Sequential(
                nn.Linear(metadata_dim, metadata_hidden_dim),
                nn.GELU(),
                nn.Linear(metadata_hidden_dim, metadata_hidden_dim),
                nn.GELU(),
            )
            total_input_dim = input_dim + metadata_hidden_dim
        else:
            self.metadata_proj = None
            total_input_dim = input_dim

        self.input_proj = nn.Linear(total_input_dim, hidden_dim)

        configs = [
            _TemporalBlockConfig(dilation=dilation_growth ** i, kernel_size=kernel_size)
            for i in range(num_layers)
        ]
        self.blocks = nn.ModuleList(
            _TemporalResidualBlock(hidden_dim, cfg, dropout) for cfg in configs
        )

        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

        self.output_scalar = output_scalar
        self.register_buffer("_eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False)

    def forward(
        self,
        sequence: Tensor,
        visibility: Tensor,
        metadata: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict the heart-rate signal from temporal features.

        Args:
            sequence: Tensor of shape ``[B, S, V]`` containing spatial features.
            visibility: Tensor of shape ``[B, S]`` containing per-frame masks.
            metadata: Optional tensor of shape ``[B, S, metadata_dim]`` with
                auxiliary data (``cx``, ``cy``, ``w``, ``h``).  When omitted the
                metadata contribution defaults to zeros.
        """

        if sequence.dim() != 3:
            raise ValueError("sequence must have shape [B, S, V]")
        if visibility.dim() != 2:
            raise ValueError("visibility must have shape [B, S]")
        if sequence.shape[:2] != visibility.shape:
            raise ValueError("sequence and visibility must align in batch and time")

        mask = (visibility > 0).to(dtype=sequence.dtype, device=sequence.device).unsqueeze(1)

        inputs = sequence
        if self.metadata_proj is not None:
            if metadata is None:
                metadata = torch.zeros(
                    sequence.shape[0],
                    sequence.shape[1],
                    self.metadata_dim,
                    dtype=sequence.dtype,
                    device=sequence.device,
                )
            else:
                if metadata.dim() != 3:
                    raise ValueError("metadata must have shape [B, S, metadata_dim]")
                if metadata.shape[:2] != sequence.shape[:2]:
                    raise ValueError("metadata must align with sequence in batch and time")
                if metadata.shape[-1] != self.metadata_dim:
                    raise ValueError(
                        f"metadata must have final dimension {self.metadata_dim}, got {metadata.shape[-1]}"
                    )
                metadata = metadata.to(dtype=sequence.dtype, device=sequence.device)

            meta_features = self.metadata_proj(metadata)
            meta_features = meta_features * mask.squeeze(1).unsqueeze(-1)
            inputs = torch.cat((sequence, meta_features), dim=-1)
        else:
            inputs = sequence

        x = self.input_proj(inputs)
        x = x.transpose(1, 2)  # [B, V, S]
        x = x * mask

        for block in self.blocks:
            x = block(x, mask)

        x = self.head(x)
        heart_rate = x.squeeze(1)
        heart_rate = heart_rate * mask.squeeze(1)

        if self.output_scalar:
            weights = mask.squeeze(1)
            eps = self._eps.to(dtype=sequence.dtype, device=sequence.device)
            denom = weights.sum(dim=1).clamp_min(eps)
            heart_rate = (heart_rate * weights).sum(dim=1) / denom

        return heart_rate


class RPPGModel(nn.Module):
    """End-to-end rPPG model that mirrors classical processing pipelines."""

    def __init__(
        self,
        *,
        spatial_encoder: Optional[RPPGSpatialEncoder] = None,
        temporal_encoder: Optional[RPPGTemporalEncoder] = None,
        return_scalar_heart_rate: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_encoder = spatial_encoder or RPPGSpatialEncoder()
        if temporal_encoder is None:
            temporal_encoder = RPPGTemporalEncoder(
                input_dim=self.spatial_encoder.feature_dim,
                output_scalar=return_scalar_heart_rate,
            )
        else:
            temporal_encoder.output_scalar = return_scalar_heart_rate

        self.temporal_encoder = temporal_encoder
        self.return_scalar_heart_rate = return_scalar_heart_rate

    def forward(self, frames: Tensor, metadata: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Run the spatial and temporal encoders sequentially.

        Args:
            frames: Tensor shaped ``[B, S, C, H, W]`` or ``[S, C, H, W]``.
            metadata: Optional tensor containing per-frame metadata structured
                as ``[visibility, cx, cy, w, h]``.  When provided the temporal
                encoder consumes the geometric components to contextualise the
                signal.
        """

        frames, metadata, squeeze_batch = _ensure_batch_first(frames, metadata)

        features, visibility = self.spatial_encoder(frames, metadata)
        if features.dim() == 2:
            features = features.unsqueeze(0)
            visibility = visibility.unsqueeze(0)

        temporal_metadata = None
        if metadata is not None:
            if metadata.dim() != 3:
                raise ValueError("metadata must have shape [B, S, 5] or [S, 5]")
            temporal_metadata = metadata[..., 1:]

        predictions = self.temporal_encoder(features, visibility, temporal_metadata)

        if squeeze_batch:
            predictions = predictions.squeeze(0)
            features = features.squeeze(0)
            visibility = visibility.squeeze(0)

        return predictions, features, visibility
