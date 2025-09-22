"""
Transformer backbone with rotary embeddings, QK-norm, and grouped query attention.
This file implements a standard Pre-LN Transformer that can be configured
as an encoder or decoder, with the specific stabilization features mentioned
in the GCP-VQVAE workplan.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from gcpvqvae.models.gcpcore import Linear


class RotaryEmbedding(nn.Module):
    """Caches and computes rotary positional embeddings."""
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_seq_len = None
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_len):
        if seq_len != self.cached_seq_len:
            self.cached_seq_len = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cached_cos = emb.cos()
            self.cached_sin = emb.sin()
        return self.cached_cos, self.cached_sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb


class Attention(nn.Module):
    """Multi-head attention with GQA, QK-norm, and RoPE."""
    def __init__(self, d_model, heads_q, heads_kv, rope: RotaryEmbedding):
        super().__init__()
        self.heads_q = heads_q
        self.heads_kv = heads_kv
        self.d_head = d_model // heads_q
        self.scale = self.d_head ** -0.5
        self.rope = rope

        self.to_q = Linear(d_model, d_model)
        self.to_k = Linear(d_model, self.d_head * heads_kv)
        self.to_v = Linear(d_model, self.d_head * heads_kv)
        self.to_out = Linear(d_model, d_model)

    def forward(self, x, mask=None):
        b, n, _, h_q, h_kv = *x.shape, self.heads_q, self.heads_kv

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h_q)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h_kv)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h_kv)

        cos, sin = self.rope(q, n)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # QK-norm
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # GQA: repeat K and V heads
        if h_q != h_kv:
            repeats = h_q // h_kv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """A Pre-LN Transformer block."""
    def __init__(self, d_model, heads_q, heads_kv, rope, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, heads_q, heads_kv, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            Linear(d_model * ffn_mult, d_model)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    """A stack of TransformerBlocks."""
    def __init__(self, d_model, depth, heads_q, heads_kv, ffn_mult=4, d_vq=256, project_out=True):
        super().__init__()
        self.d_model = d_model
        self.rope = RotaryEmbedding(d_model // heads_q)
        self.proj_in = Linear(d_vq, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, heads_q, heads_kv, self.rope, ffn_mult)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = Linear(d_model, d_vq) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        x = self.proj_in(x)

        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') & rearrange(mask, 'b j -> b 1 1 j')

        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        x = self.norm_out(x)
        return self.proj_out(x)
