"""End-to-end differential pytest: fusion preserves numerics (Phase 2, Task 11).

Builds the :class:`~forge_torch.conftest.FusedMLP` model, runs only
the ``fuse`` xDSL pass (via ``forge-opt --pass-pipeline='fuse'``) on
its ``forge.graph`` MLIR text through :mod:`subprocess`, then runs
**both** the original MLIR (via :func:`~forge_torch.backend.compile`)
and the post-fusion MLIR through the backend hook
(:class:`~forge_torch.backend.CompiledFusedMLP`). Asserts that
:func:`~forge_torch.validation.validate` passes for both, for the 3
existing shapes (spec §19).

The ``fuse`` pass collapses the canonical FusedMLP triple
``forge.tensor.gemm + forge.tensor.broadcast_add + forge.tensor.fused_epilogue``
into a ``gemm + fused_epilogue-with-bias`` pair (3 → 2 ops; the 3 → 1
collapse needs a new ``fused_gemm_epilogue`` op that is Phase-3
dialect work — see :mod:`compiler.lib.Transforms.Fuse` for the full
rationale). The pattern is structurally a no-op for the FusedMLP
``forge.graph`` IR (the pass only matches ``forge.tensor.*`` ops), so
on a non-lowered IR the test pins the contract that the pipeline
remains verifier-clean and numerically identical after fuse.
"""

from __future__ import annotations

import subprocess

import pytest
import torch
from forge_torch.backend import CompiledFusedMLP, compile
from forge_torch.conftest import FusedMLP
from forge_torch.validation import validate

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
            "/home/mei/projects/forgestack/.venv/bin/python3",
            "/home/mei/projects/forgestack/compiler/tools/forge-opt/forge_opt.py",
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
def test_fusion_preserves_numerics(B: int, S: int, K: int, N: int) -> None:
    """Differential: post-fusion MLIR validates against eager for all 3 shapes.

    Both paths — the original ``compile()`` call and the post-fusion
    ``CompiledFusedMLP`` wrapper around the optimized MLIR — must
    produce output that :func:`validate` accepts against the eager
    reference. The fuse pass must therefore leave numerics identical
    (the algebraic identity ``gelu(a + b) == fused_epilogue(gemm + b)``
    holds regardless of the rewrite form).
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

    # Path 2 — fusion-pass output, fed back through the same backend hook.
    optimized_mlir = _run_pass(compiled.mlir_text, "fuse")
    optimized_compiled = CompiledFusedMLP(model, [x, w, b], optimized_mlir)
    optimized_out = optimized_compiled(x, w, b)
    result_optimized = validate(optimized_out, eager_out)
    assert result_optimized.passed, (
        f"post-fusion MLIR failed validation: {result_optimized}\nerrors: {result_optimized.errors}"
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
