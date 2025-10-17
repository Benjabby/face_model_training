from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class _SinePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding reused for any sequence length."""

    def __init__(self, embed_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)].to(x.dtype)


class _TemporalBlock(nn.Module):
    """A single transformer block with explicit attention masking."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        x = self.ff_norm(x + self.ff(x))
        return x


class TemporalEncoder(nn.Module):
    """Encode temporal sequences into heart-rate predictions."""

    def __init__(
        self,
        feature_dim: int = 128,
        embed_dim: Optional[int] = None,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        output_scalar: bool = True,
    ) -> None:
        super().__init__()
        embed_dim = embed_dim or feature_dim
        ff_hidden_dim = ff_hidden_dim or embed_dim * 4

        self.input_proj = (
            nn.Identity() if feature_dim == embed_dim else nn.Linear(feature_dim, embed_dim)
        )
        self.positional_encoding = _SinePositionalEncoding(embed_dim, max_seq_len)

        self.layers = nn.ModuleList(
            _TemporalBlock(embed_dim, num_heads, dropout, ff_hidden_dim)
            for _ in range(num_layers)
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        self.num_heads = num_heads
        self.output_scalar = output_scalar

    def _build_attn_mask(self, visibility: Tensor) -> Tensor:
        """Construct a visibility-aware attention mask for all heads."""

        if visibility.dim() != 2:
            raise ValueError("visibility must have shape [B, S]")

        mask = visibility > 0
        attn = mask.unsqueeze(1) & mask.unsqueeze(2)
        attn_bias = torch.zeros(
            attn.shape, dtype=torch.float32, device=visibility.device
        ).masked_fill(~attn, float("-inf"))

        # Rows with no visible positions lead to -inf masking; neutralize them.
        invalid_rows = (~attn).all(dim=-1, keepdim=True)
        if invalid_rows.any():
            attn_bias = attn_bias.masked_fill(invalid_rows, 0.0)

        attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn_bias = attn_bias.view(-1, attn_bias.size(-2), attn_bias.size(-1))
        return attn_bias

    def forward(self, sequence: Tensor, visibility: Tensor) -> Tensor:
        """Predict heart-rate signals from temporal embeddings."""

        if sequence.dim() != 3:
            raise ValueError("sequence must have shape [B, S, V]")
        if visibility.shape != sequence.shape[:2]:
            raise ValueError("visibility must align with sequence length")

        if visibility.device != sequence.device:
            visibility = visibility.to(sequence.device)

        x = self.input_proj(sequence)
        x = x + self.positional_encoding(x)

        attn_mask = self._build_attn_mask(visibility).to(x.dtype)

        for layer in self.layers:
            x = layer(x, attn_mask)

        heart_rate = self.output_head(x).squeeze(-1)

        if self.output_scalar:
            weights = visibility.to(dtype=heart_rate.dtype)
            weights = weights.clamp_min(0.0)
            denom = weights.sum(dim=1).clamp_min(torch.finfo(heart_rate.dtype).eps)
            heart_rate = (heart_rate * weights).sum(dim=1) / denom

        return heart_rate
