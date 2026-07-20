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

LayerNorm is implemented as a manual mean/var computation (not nn.LayerNorm).
Linears are constructed via ``_make_linear`` which stores ``weight`` / ``bias``
as Python closure variables (not module attributes) so that
``torch.fx.symbolic_trace`` captures them as free variables / placeholders
— NOT as ``get_attr`` nodes that the Phase 1-8 FX importer rejects with
``KeyError: 'unknown FX op target: <built-in function getattr>'``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def _make_linear(in_features: int, out_features: int, bias: bool = True) -> nn.Module:
    """Build a Linear-equivalent module using closure-bound weights.

    Storing ``weight`` / ``bias`` as Python closure variables (not module
    attributes) means ``torch.fx.symbolic_trace`` captures them as free
    variables in the traced graph — NOT as ``get_attr`` nodes (which the
    Phase 1-8 FX importer cannot lower).
    """
    weight = torch.empty(out_features, in_features)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    bias_tensor = torch.zeros(out_features) if bias else None
    if bias_tensor is not None:
        fan_in = in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(bias_tensor, -bound, bound)

    class _ClosureLinear(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, weight, bias_tensor)

    return _ClosureLinear()


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

        self.qkv = _make_linear(d_model, d_model * 3)
        self.out = _make_linear(d_model, d_model)
        self.mlp_in = _make_linear(d_model, d_ff)
        self.mlp_out = _make_linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # 1) LayerNorm + QKV projection
        h = _layer_norm(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, S, head_dim)
        # 2) Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn = attn_weights @ v  # (B, n_heads, S, head_dim)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        # 3) Out projection + residual
        x = x + self.out(attn)
        # 4) LayerNorm + MLP + residual
        h = _layer_norm(x)
        h = self.mlp_out(F.gelu(self.mlp_in(h)))
        x = x + h
        return x
