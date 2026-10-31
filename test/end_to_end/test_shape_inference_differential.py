"""End-to-end differential pytest: shape inference preserves numerics (Phase 2, Task 10).

Builds the :class:`~forge_torch.conftest.FusedMLP` model, runs only
the ``shape-inference`` xDSL pass (via ``forge-opt
--pass-pipeline='shape-inference'``) on its ``forge.graph`` MLIR text
through :mod:`subprocess`, then runs **both** the original MLIR (via
:func:`~forge_torch.backend.compile`) and the post-shape-inference MLIR
through the backend hook (:class:`~forge_torch.backend.CompiledFusedMLP`).
Asserts that :func:`~forge_torch.validation.validate` passes for both,
for the 3 existing shapes (spec §19).

Shape inference must be observably a no-op for numerics: it rewrites
op result *types* (replacing symbolic-dim ``StringAttr`` with concrete
``IntAttr`` where the dialect rules make the shape computable) without
touching operands or rewiring the dataflow graph. The differential
test pins down that contract end-to-end so a future shape-inference
regression that e.g. dropped a dim is caught immediately.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from forge_torch.backend import CompiledFusedMLP, compile
from forge_torch.conftest import FusedMLP
from forge_torch.validation import validate

REPO = Path(__file__).resolve().parents[2]

SHAPES = [
    # (B, S, K, N) — Phase 1 canonical (spec §19)
    (2, 512, 4096, 11008),
    # Small smoke
    (1, 128, 2048, 4096),
    # Tiny smoke
    (4, 64, 512, 1024),
]


def _run_pass(mlir_text: str, pass_name: str) -> str:
    """Run ``forge-opt --pass-pipeline=<pass_name>`` over ``mlir_text``.

    Used to exercise a single xDSL pass (canonicalize / shape-inference
    / fuse) in isolation. The driver is invoked via :mod:`subprocess`
    so the test exercises the same entry point end-users will hit from
    the CLI — any import / sys.path / dialect-discovery regression in
    the driver is caught here too.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "compiler" / "tools" / "forge-opt" / "forge_opt.py"),
            "-",
            "--action=optimize",
            f"--pass-pipeline={pass_name}",
        ],
        input=mlir_text,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


@pytest.mark.parametrize("B,S,K,N", SHAPES)
def test_shape_inference_preserves_numerics(B: int, S: int, K: int, N: int) -> None:
    """Differential: post-shape-inference MLIR validates against eager for all 3 shapes.

    Both paths — the original ``compile()`` call and the
    post-shape-inference ``CompiledFusedMLP`` wrapper around the
    optimized MLIR — must produce output that :func:`validate` accepts
    against the eager reference. The shape-inference pass must
    therefore rewrite only result types, leaving numerics untouched.
    """
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(B, S, K, dtype=torch.float32)
    w = torch.randn(K, N, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)

    eager_out = model(x, w, b)

    # Path 1 — original (un-optimized) MLIR via the existing ``compile()`` hook.
    compiled = compile(model, [x, w, b])
    compiled_out = compiled(x, w, b)
    result_original = validate(compiled_out, eager_out)
    assert result_original.passed, (
        f"original MLIR failed validation: {result_original}\nerrors: {result_original.errors}"
    )

    # Path 2 — shape-inference-pass output, fed back through the same backend hook.
    optimized_mlir = _run_pass(compiled.mlir_text, "shape-inference")
    optimized_compiled = CompiledFusedMLP(model, [x, w, b], optimized_mlir)
    optimized_out = optimized_compiled(x, w, b)
    result_optimized = validate(optimized_out, eager_out)
    assert result_optimized.passed, (
        f"post-shape-inference MLIR failed validation: {result_optimized}\n"
        f"errors: {result_optimized.errors}"
    )

    # Both must agree on the structural preconditions (shape / dtype /
    # NaN / Inf) so the tolerance check below is meaningful.
    assert result_original.shape_ok and result_original.dtype_ok
    assert result_optimized.shape_ok and result_optimized.dtype_ok
    assert not result_optimized.has_nan and not result_optimized.has_inf

    # fp32 tolerance: atol=1e-4 per spec §15 (matches test_fused_mlp_e2e.py).
    assert result_optimized.max_abs_err < 1e-4, (
        f"max_abs_err={result_optimized.max_abs_err} exceeds fp32 atol=1e-4"
    )
