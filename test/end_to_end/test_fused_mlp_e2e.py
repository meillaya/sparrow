"""End-to-end FusedMLP differential pytest (Phase 1, Task 13; Phase 2, Task 20).

Builds the FusedMLP model, runs eager, runs through
the ``forge_torch.compile`` wrapper, asserts the validation result
passes for at least 3 random shapes (spec §19).

Phase 2 / Wave 5 (Task 20) extends the shape coverage to 6 by adding
three stress shapes: ``[8, 1024, 8192, 16384]`` (heavy square matmul),
``[16, 2048, 1024, 4096]`` (long-sequence batch), ``[1, 1, 512, 2048]``
(singleton batch — exercises the per-axis cases the canonical
``[2, 512, 4096, 11008]`` doesn't).

The shipped scope keeps the compiled path delegating to the eager
kernel, so the differential test confirms the *pipeline* — FX capture,
MLIR import, lowering, ExecutionPlan build, scheduler dispatch —
produces a tensor that numerically matches eager. The C++ scheduler +
provider are exercised independently via ``runtime/tests/test_scheduler``
and ``runtime/tests/test_triton_provider`` (Tasks 11/12).
"""

from __future__ import annotations

import pytest
import torch

from forge_torch.backend import BACKEND, compile
from forge_torch.conftest import FusedMLP
from forge_torch.validation import validate

SHAPES = [
    # (B, S, K, N) — Phase 1 canonical (spec §19)
    (2, 512, 4096, 11008),
    # Small smoke
    (1, 128, 2048, 4096),
    # Tiny smoke
    (4, 64, 512, 1024),
    # Wave 5 stress shapes (Task 20)
    (8, 1024, 8192, 16384),
    (16, 2048, 1024, 4096),
    (1, 1, 512, 2048),
]


@pytest.mark.parametrize("B,S,K,N", SHAPES)
def test_fused_mlp_compiled_matches_eager(B: int, S: int, K: int, N: int) -> None:
    """Differential: compiled output matches eager within fp32 tolerance.

    The compiled callable goes through the full Phase-1 pipeline
    (FX → MLIR → ExecutionPlan → scheduler → provider stub → eager
    kernel) and is asserted equivalent to the eager reference using
    :func:`forge_torch.validation.validate`.
    """
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(B, S, K, dtype=torch.float32)
    w = torch.randn(K, N, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)

    eager_out = model(x, w, b)
    compiled = compile(model, [x, w, b])
    compiled_out = compiled(x, w, b)

    result = validate(compiled_out, eager_out)
    assert result.passed, f"validation failed: {result}\nerrors: {result.errors}"
    assert result.shape_ok
    assert result.dtype_ok
    assert not result.has_nan
    assert not result.has_inf
    # fp32 tolerance: atol=1e-4 per spec §15.
    assert result.max_abs_err < 1e-4, f"max_abs_err={result.max_abs_err} exceeds fp32 atol=1e-4"


def test_fused_mlp_smoke_compile_callable_runs() -> None:
    """Smoke: ``compile()`` returns a callable that accepts the right args."""
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(1, 8, 16, dtype=torch.float32)
    w = torch.randn(16, 32, dtype=torch.float32)
    b = torch.randn(32, dtype=torch.float32)
    compiled = compile(model, [x, w, b])
    out = compiled(x, w, b)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 8, 32)
    # Sanity: confirms FX → forge.graph import ran during compile().
    assert "forge.graph" in compiled.mlir_text
    # Sanity: the backend name matches what ``torch.compile(backend=...)``
    # would receive.
    assert BACKEND == "sparrow"
