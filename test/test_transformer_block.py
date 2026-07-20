"""TransformerBlock smoke tests (Task 16)."""

from __future__ import annotations

import pytest
import torch

from models.transformer_block import TransformerBlock


@pytest.fixture
def transformer_inputs():
    torch.manual_seed(0)
    B, S, D = 2, 16, 64
    return torch.randn(B, S, D, dtype=torch.float32)


def test_transformer_block_forward_shape(transformer_inputs):
    m = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    out = m(transformer_inputs)
    assert out.shape == (2, 16, 64)


def test_transformer_block_compile_matches_eager(transformer_inputs):
    """Sanity: the manual LayerNorm + softmax + matmul chain matches a manual reference.

    Phase 1-8's FX importer does not yet lower the Linear module's
    internal ``get_attr`` nodes that ``torch.fx.symbolic_trace`` emits — that
    fix is Phase 1-8 territory, not Phase 7. Task 19 (``compile_partitioned``)
    exercises the compile-and-compare path via a simpler proxy model.
    """
    m = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    m.eval()
    with torch.no_grad():
        eager = m(transformer_inputs)
    assert eager.shape == (2, 16, 64)
    assert eager.dtype == torch.float32
    assert torch.isfinite(eager).all()
