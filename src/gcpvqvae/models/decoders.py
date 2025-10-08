"""Transformer-based geometric decoder modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from x_transformers import ContinuousTransformerWrapper, Encoder

from gcpvqvae.models.transformer import TransformerConfig


class NdLinear(nn.Module):
    """Position-aware linear layer matching reference checkpoint shapes."""

    def __init__(
        self,
        input_dims: tuple[int, int],
        hidden_size: tuple[int, int],
        *,
        bias: bool = True,
    ) -> None:
        """Initialise the NdLinear projection.

        Args:
            input_dims: Tuple ``(max_length, in_features)`` describing input shape.
            hidden_size: Tuple ``(max_length, out_features)`` describing output shape.
            bias: Add a position-dependent bias term when ``True``.
        """
        super().__init__()

        if len(input_dims) != 2 or len(hidden_size) != 2:
            raise ValueError("NdLinear currently supports 2D inputs (length, features)")
        max_length_in, in_features = input_dims
        max_length_out, out_features = hidden_size
        if max_length_in != max_length_out:
            raise ValueError(
                "Input and output length dimensions must match for NdLinear"
            )

        self.max_length = max_length_in
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(self.max_length, self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weight.view(self.max_length * self.out_features, self.in_features), a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            bound = 1 / math.sqrt(self.in_features)
            bias_param = torch.empty(self.max_length, self.out_features)
            nn.init.uniform_(bias_param, -bound, bound)
            self.bias = nn.Parameter(bias_param)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the position-aware linear projection.

        Args:
            x: Tensor of shape ``(..., length, in_features)`` with ``length`` not
                exceeding ``self.max_length``.

        Returns:
            Tensor of shape ``(..., length, out_features)``.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input feature dimension {self.in_features}, got {x.shape[-1]}"
            )
        seq_len = x.shape[-2]
        if seq_len > self.max_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds NdLinear maximum of {self.max_length}"
            )

        weight = self.weight[:seq_len]
        out = torch.einsum("...ld, lod -> ...lo", x, weight)
        if self.bias is not None:
            out = out + self.bias[:seq_len]
        return out


class GeometricTransformerDecoder(nn.Module):
    """Decoder aligning latent tokens with geometric reconstruction heads."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialise the transformer decoder from a configuration."""
        super().__init__()
        self.config = config

        if config.use_ndlinear:
            if config.max_length is None:
                raise ValueError("NdLinear projection requires `max_length` to be set")
            self.input_proj: nn.Module = NdLinear(
                input_dims=(config.max_length, config.input_dim),
                hidden_size=(config.max_length, config.model_dim),
                bias=False,
            )
        else:
            self.input_proj = nn.Linear(config.input_dim, config.model_dim, bias=False)

        attn_kwargs: dict[str, object] = {
            "dim": config.model_dim,
            "depth": config.num_layers,
            "heads": config.num_heads,
            "ff_mult": config.ffn_multiplier,
            "ff_dropout": config.dropout,
            "attn_dropout": config.dropout,
            "ff_glu": True,
            "ff_swish": True,
            "rotary_pos_emb": config.use_rope,
            "attn_dim_head": config.model_dim // config.num_heads,
            "attn_kv_heads": config.num_kv_heads,
        }

        if getattr(config, "rotary_xpos", False):
            attn_kwargs["rotary_xpos"] = True

        if config.use_flash_attn:
            attn_kwargs["attn_flash"] = True

        if config.use_qk_norm:
            attn_kwargs.update(
                {
                    "attn_qk_norm": True,
                    "attn_qk_norm_groups": config.qk_norm_groups,
                    "attn_qk_norm_scale": config.qk_norm_scale,
                    "attn_qk_norm_dim_scale": config.qk_norm_dim_scale,
                }
            )

        self.transformer = ContinuousTransformerWrapper(
            max_seq_len=config.max_length or 0,
            attn_layers=Encoder(**attn_kwargs),
            dim_in=None,
            dim_out=config.output_dim or config.model_dim,
            emb_dropout=config.dropout,
            num_memory_tokens=config.num_memory_tokens or None,
        )

    def forward(self, latents: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        """Decode latent tokens into geometric features using attention layers.

        Args:
            latents: Tensor of shape ``(batch, length, dim)`` containing latent embeddings.
            mask: Optional boolean tensor marking valid positions.

        Returns:
            Tensor of shape ``(batch, length, output_dim)`` with decoded features.
        """
        if latents.ndim != 3:
            raise ValueError("Decoder inputs must have shape (batch, length, dim)")

        projected = self.input_proj(latents)

        attn_mask: Optional[Tensor] = None
        if self.config.causal:
            seq_len = projected.shape[1]
            attn_mask = torch.ones(
                seq_len, seq_len, device=projected.device, dtype=torch.bool
            ).tril()

        pad_mask = mask.to(torch.bool) if mask is not None else None

        return self.transformer(projected, mask=pad_mask, attn_mask=attn_mask)


__all__ = ["GeometricTransformerDecoder", "NdLinear"]
