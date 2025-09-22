"""Transformer backbone with rotary embeddings and grouped-query attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    batch, length, dim = x.shape
    head_dim = dim // num_heads
    return x.view(batch, length, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_dim)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embeddings require an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        return cos, sin


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        dropout: float = 0.0,
        rope: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be a multiple of num_kv_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.rope = rope

    def _repeat_kv(self, tensor: Tensor) -> Tensor:
        if self.groups == 1:
            return tensor
        return tensor.unsqueeze(2).repeat(1, 1, self.groups, 1, 1).reshape(
            tensor.size(0), self.num_heads, tensor.size(-2), self.head_dim
        )

    def forward(self, x: Tensor, *, attn_mask: Optional[Tensor] = None) -> Tensor:
        batch, length, _ = x.shape

        q = _split_heads(self.q_proj(x), self.num_heads)
        k = _split_heads(self.k_proj(x), self.num_kv_heads)
        v = _split_heads(self.v_proj(x), self.num_kv_heads)

        if self.rope is not None:
            cos, sin = self.rope(length, device=x.device, dtype=x.dtype)
            q, k = apply_rotary(q, k, cos, sin)

        q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-6)
        k = k / torch.clamp(torch.linalg.norm(k, dim=-1, keepdim=True), min=1e-6)

        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)

        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v_full)
        output = _merge_heads(output)
        return self.proj_dropout(self.out_proj(output))


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        dropout: float,
        ffn_multiplier: float,
        rope: Optional[RotaryEmbedding],
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim,
            num_heads,
            num_kv_heads,
            dropout=dropout,
            rope=rope,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_multiplier, dropout)

    def forward(self, x: Tensor, *, attn_mask: Optional[Tensor]) -> Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass
class TransformerConfig:
    input_dim: int
    model_dim: int = 1024
    output_dim: Optional[int] = None
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 3
    dropout: float = 0.0
    ffn_multiplier: float = 4.0
    use_rope: bool = True


class GCPTokensTransformer(nn.Module):
    """Pre-LN Transformer stack tailored to the GCP-VQVAE architecture."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.model_dim, bias=False)
        self.output_proj = nn.Linear(
            config.model_dim, config.output_dim or config.model_dim, bias=False
        )

        rope = RotaryEmbedding(config.model_dim // config.num_heads) if config.use_rope else None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.model_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                    rope=rope,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.model_dim)

    def forward(self, x: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Inputs must have shape (batch, length, features)")

        attn_mask: Optional[Tensor]
        if mask is None:
            attn_mask = None
        else:
            mask = mask.to(torch.bool)
            attn_mask = (~mask).unsqueeze(1).unsqueeze(2)

        hidden = self.input_proj(x)
        for block in self.blocks:
            hidden = block(hidden, attn_mask=attn_mask)

        hidden = self.final_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output_proj(hidden)


__all__ = ["GCPTokensTransformer", "TransformerConfig"]
