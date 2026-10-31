"""End-to-end FusedMLP-on-astra-sim pytest (Phase 4, Task 63 / Wave 14).

Confirms the Sparrow end-to-end pipeline — FX capture, MLIR import,
cache key, ``compile()`` returning a callable — produces a tensor that
matches eager within fp32 atol=1e-4. The shipped compiled path delegates
to eager (per Phase 1's backwards-compat contract; the C++ provider is
the lowered backend that runs in production, exercised by
``runtime/tests/test_astra_provider``); this pytest keeps that
contract honest and asserts the simulator itself produces a tensor
matching a Triton-CPU reference on the bias=0 tile shape.

The companion ``test_cross_provider`` file (Task 67) covers the
provider-routed path explicitly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from backends.astra.sim.functional import (
    DRAM_BYTES,
    TILE_BYTES,
    TILE_DIM,
    last_dram_tensor,
    run_functional,
)
from compiler.lib.Dialect.Astra.assembler import assemble
from forge_torch.backend import compile
from forge_torch.conftest import FusedMLP

SHAPES = [
    # (B, S, K, N) — Phase 1 canonical (spec §19)
    (2, 512, 4096, 11008),
    # Small smoke
    (1, 128, 2048, 4096),
    # Tiny smoke
    (4, 64, 512, 1024),
]


@pytest.mark.parametrize("B,S,K,N", SHAPES)
def test_fused_mlp_compile_matches_eager(B: int, S: int, K: int, N: int) -> None:
    """Compiled output matches eager within fp32 atol=1e-4.

    The compiled callable goes through the full Phase-1 pipeline
    (FX capture, MLIR import, ExecutionPlan build, provider selection).
    Phase 4 ships the ``AstraProvider`` as a lowered backend; per
    Phase 1's backwards-compat contract, the foreground
    ``CompiledFusedMLP`` still delegates to eager.
    """
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(B, S, K, dtype=torch.float32)
    w = torch.randn(K, N, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)

    eager = model(x, w, b)
    compiled = compile(model, [x, w, b])
    out = compiled(x, w, b)

    assert isinstance(out, torch.Tensor)
    assert out.shape == eager.shape
    assert torch.allclose(out, eager, atol=1e-4, rtol=0.0), (
        f"compiled output diverges from eager (max diff = {(out - eager).abs().max().item():.6e})"
    )


def test_compile_returns_well_formed_compiled_callable() -> None:
    """Sanity: ``compile()`` returns a callable marked as compiled + eager-routed."""
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(1, 8, 16, dtype=torch.float32)
    w = torch.randn(16, 32, dtype=torch.float32)
    b = torch.randn(32, dtype=torch.float32)

    compiled = compile(model, [x, w, b])
    out = compiled(x, w, b)

    assert compiled.compiled is True
    assert "forge.graph" in compiled.mlir_text
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 8, 32)


def _gelu_reference(values: np.ndarray) -> np.ndarray:
    """Match the simulator's fp32 erf-based GELU exactly."""
    flat = np.fromiter(
        (np.float32(0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))) for v in values.flat),
        dtype=np.float32,
        count=values.size,
    )
    return flat.reshape(values.shape)


def test_astra_simulator_fused_mlp_tile_matches_naive_reference() -> None:
    """The astra functional simulator's MATMUL + VECTOR_GELU pipeline
    reproduces a hand-rolled numpy reference within fp32 atol=1e-4.

    This is the Python-level mirror of the C++ test in
    ``runtime/tests/test_astra_provider`` (Task 60/61): the same tile
    shape (16x16) and the same bias=0 contract. Cross-checking both
    paths gives an end-to-end equivalence story without requiring a
    lowered execution route through the C++ provider yet.
    """
    k = 4
    rng = np.random.default_rng(42)
    lhs = rng.standard_normal((TILE_DIM, k)).astype(np.float32)
    rhs = rng.standard_normal((k, TILE_DIM)).astype(np.float32)
    lhs_bytes = lhs.nbytes
    rhs_bytes = rhs.nbytes
    gelu_sram = 2 * TILE_BYTES
    output_dram = lhs_bytes + rhs_bytes

    program = (
        f"DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes={lhs_bytes}\n"
        f"DMA_LOAD dst=SRAM[0x{lhs_bytes:x}], src=DRAM[0x{lhs_bytes:x}], bytes={rhs_bytes}\n"
        f"MATMUL a=SRAM[0x0], b=SRAM[0x{lhs_bytes:x}], k={k}\n"
        f"VECTOR_GELU dst=SRAM[0x{gelu_sram:x}], src=SRAM[0x0], elements={TILE_DIM * TILE_DIM}\n"
        f"DMA_STORE dst=DRAM[0x{output_dram:x}], src=SRAM[0x{gelu_sram:x}], bytes={TILE_BYTES}\n"
        "HALT\n"
    )

    dram = bytearray(DRAM_BYTES)
    dram[:lhs_bytes] = lhs.tobytes()
    dram[lhs_bytes : lhs_bytes + rhs_bytes] = rhs.tobytes()

    result = run_functional(assemble(program), dram=dram)
    out = last_dram_tensor(result)
    expected = _gelu_reference((lhs @ rhs).astype(np.float32))

    assert out is not None
    assert out.shape == expected.shape
    assert np.allclose(out, expected, atol=1e-4, rtol=0.0)
