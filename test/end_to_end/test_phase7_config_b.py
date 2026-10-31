"""Config B fails closed until the Astra lowering lands (M4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from forge_torch.compile_partitioned import compile_partitioned


class _MatmulThenGelu(torch.nn.Module):
    def forward(self, x, w, b):
        return F.gelu(x @ w + b)


def test_config_b_raises_without_astra_lowering() -> None:
    model = _MatmulThenGelu()
    inputs = [torch.randn(4, 8), torch.randn(8, 16), torch.randn(16)]
    with pytest.raises(NotImplementedError, match="Astra"):
        compile_partitioned(model, inputs, config="B")
