"""Pytest fixtures for ForgeStack frontend tests (Task 8)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class FusedMLP(nn.Module):
    """Phase 1 vertical slice: ``x @ w + b`` followed by GELU."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x @ weight + bias)


@pytest.fixture
def fixed_seed() -> int:
    torch.manual_seed(0)
    return 0


@pytest.fixture
def fused_mlp() -> FusedMLP:
    return FusedMLP()


@pytest.fixture
def fused_mlp_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape ``[2, 512, 4096] x [4096, 11008] + [11008]`` — Phase 1 canonical shape."""
    x = torch.randn(2, 512, 4096, dtype=torch.float32)
    w = torch.randn(4096, 11008, dtype=torch.float32)
    b = torch.randn(11008, dtype=torch.float32)
    return x, w, b
