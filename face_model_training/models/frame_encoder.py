from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class FrameEncoder(nn.Module):
    """Encode per-frame face crops with metadata-aware transformers."""

    def __init__(
        self,
        in_channels: int = 6,
        image_size: int = 112,
        patch_size: int = 16,
        embed_dim: int = 128,
        image_layers: int = 2,
        meta_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.embed_dim = embed_dim
        patches_per_dim = image_size // patch_size
        self.num_patches = patches_per_dim * patches_per_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.image_transformer = nn.TransformerEncoder(encoder_layer, num_layers=image_layers)

        meta_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.meta_transformer = nn.TransformerEncoder(meta_layer, num_layers=meta_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.meta_pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

        self.meta_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, embed_dim),
        )

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.meta_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.meta_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, frames: Tensor, metadata: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode frames and metadata.

        Args:
            frames: Either ``[B, C, 112, 112]`` or ``[B, S, C, 112, 112]`` tensors
            with ``C`` matching the ``in_channels`` value provided at
            initialization.
            metadata: Either ``[B, 5]`` or ``[B, S, 5]`` tensors aligned with ``frames``.

        Returns:
            A tuple containing the encoded features with shape ``[B, S?, embed_dim]`` and
            the visibility mask derived from ``metadata[..., 0]``.
        """
        single_frame = frames.dim() == 4
        if single_frame:
            frames = frames.unsqueeze(1)
            metadata = metadata.unsqueeze(1)

        if frames.dim() != 5:
            raise ValueError("frames must be 4D or 5D tensors")
        if metadata.dim() != 3:
            raise ValueError("metadata must align with frames")
        if frames.shape[:2] != metadata.shape[:2]:
            raise ValueError("frames and metadata batch dimensions must match")

        if metadata.shape[-1] != 5:
            raise ValueError("metadata must have 5 features per frame")

        batch, seq_len = frames.shape[:2]
        flat_frames = frames.reshape(batch * seq_len, *frames.shape[2:])
        flat_metadata = metadata.reshape(batch * seq_len, metadata.shape[-1])

        tokens = self.patch_embed(flat_frames)
        tokens = tokens.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.image_transformer(tokens)
        tokens = self.pre_norm(tokens)

        frame_tokens = tokens[:, 0]
        meta_tokens = self.meta_proj(flat_metadata)

        joint_tokens = torch.stack((frame_tokens, meta_tokens), dim=1)
        joint_tokens = joint_tokens + self.meta_pos_embed[:, : joint_tokens.size(1)]
        joint_tokens = self.meta_transformer(joint_tokens)
        joint_tokens = self.post_norm(joint_tokens)

        encoded = joint_tokens[:, 0]
        encoded = encoded.view(batch, seq_len, self.embed_dim)

        visibility = metadata[..., 0]

        if single_frame:
            encoded = encoded.squeeze(1)
            visibility = visibility.squeeze(1)

        return encoded, visibility
