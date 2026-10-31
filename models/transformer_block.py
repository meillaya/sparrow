"""Small transformer block per spec §19 (CPU-only, defaults for Phase 7 demo).

Architecture:
  LayerNorm(d_model)
  → Linear(d_model, d_model*3)   # QKV projection
  → reshape + split heads (n_heads, head_dim = d_model // n_heads)
  → scaled_dot_product_attention = softmax(QK^T / sqrt(d_head)) @ V
  → reshape
  → Linear(d_model, d_model)      # Out projection
  → + x                          # residual
  → LayerNorm(d_model)
  → MLP: Linear(d_model, d_ff) → gelu → Linear(d_ff, d_model)
  → + x                          # residual

The parameters are ordinary ``nn.Linear`` / ``nn.LayerNorm`` modules. The
FX importer lifts them into execution-plan inputs, so the compiled path
does not depend on closure-bound weights. Layer normalisation is invoked
through ``F.layer_norm`` (rather than the module's ``__call__``) so the
graph exposes one ``aten.layer_norm`` op instead of a submodule call.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Layer norm over the last dimension, with explicit gamma and beta."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_ff = d_ff

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.mlp_in = nn.Linear(d_model, d_ff)
        self.mlp_out = nn.Linear(d_ff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # A rank-1 tensor (not a Python float) so the attention scale is a
        # first-class broadcast operand in the captured graph.
        self.register_buffer("attn_scale", torch.tensor([1.0 / math.sqrt(self.head_dim)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # 1) LayerNorm + QKV projection
        h = _layer_norm(x, self.ln1.weight, self.ln1.bias, self.ln1.eps)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, S, head_dim)
        # 2) Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.attn_scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn = attn_weights @ v  # (B, n_heads, S, head_dim)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        # 3) Out projection + residual
        x = x + self.out(attn)
        # 4) LayerNorm + MLP + residual
        h = _layer_norm(x, self.ln2.weight, self.ln2.bias, self.ln2.eps)
        h = self.mlp_out(F.gelu(self.mlp_in(h)))
        x = x + h
        return x
