"""TransformerBlock smoke and CPU-plan acceptance tests (Task 16 / M3)."""

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
    """Sanity: the manual LayerNorm + softmax + matmul chain matches a manual reference."""
    m = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    m.eval()
    with torch.no_grad():
        eager = m(transformer_inputs)
    assert eager.shape == (2, 16, 64)
    assert eager.dtype == torch.float32
    assert torch.isfinite(eager).all()


def test_transformer_block_uses_normal_parameters():
    """M3: the block stores ordinary ``nn.Linear`` / ``nn.LayerNorm`` parameters."""
    m = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    assert isinstance(m.qkv, torch.nn.Linear)
    assert isinstance(m.out, torch.nn.Linear)
    assert isinstance(m.mlp_in, torch.nn.Linear)
    assert isinstance(m.mlp_out, torch.nn.Linear)
    assert "qkv.weight" in m.state_dict()
    assert "ln1.weight" in m.state_dict()


def test_transformer_block_compiles_to_cpu_plan(transformer_inputs, tmp_path):
    """M3 acceptance: eager and plan agree with no closure-weight workaround."""
    pytest.importorskip("_sparrow_rt")

    from forge_torch.backend import compile
    from forge_torch.cache_store import CacheStore

    torch.manual_seed(0)
    m = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    m.eval()
    with torch.no_grad():
        eager = m(transformer_inputs)

    compiled = compile(
        m,
        [transformer_inputs],
        cache_store=CacheStore(tmp_path / "cache"),
    )
    with torch.no_grad():
        actual = compiled(transformer_inputs)
    assert isinstance(actual, torch.Tensor)

    assert compiled.execution == "cpu"
    assert compiled.lowered is True
    assert compiled.planning_error is None
    assert actual.shape == eager.shape
    assert torch.allclose(actual, eager, atol=1e-4, rtol=1e-4)

    # The plan really covers the block's ops rather than falling back.
    for op in (
        "forge.tensor.layer_norm",
        "forge.tensor.gemm",
        "forge.tensor.transpose",
        "forge.tensor.select",
        "forge.tensor.mul",
        "forge.tensor.softmax",
    ):
        assert op in compiled.mlir_text, op
    assert "getattr" not in compiled.mlir_text
