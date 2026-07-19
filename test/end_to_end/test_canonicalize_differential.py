"""End-to-end differential pytest: canonicalize preserves numerics (Phase 2, Task 9).

Builds the :class:`~forge_torch.conftest.FusedMLP` model, runs the
``forge-opt --action=optimize`` driver on its ``forge.graph`` MLIR text
via :mod:`subprocess`, then runs **both** the original (un-optimized)
MLIR (via the existing :func:`~forge_torch.backend.compile` path) and
the post-canonicalize MLIR through the backend hook
(:class:`~forge_torch.backend.CompiledFusedMLP`). Asserts that
:func:`~forge_torch.validation.validate` passes for both, for the 3
existing shapes ``[2,512,4096,11008]``, ``[1,128,2048,4096]``,
``[4,64,512,1024]`` (spec §19).

The compiled callable in the shipped Phase-1 scope delegates to the
eager kernel (see :mod:`forge_torch.backend`), so this differential
test confirms the *canonicalize pipeline* — pattern-rewrite walker,
verifier re-runs, textual print-back — produces MLIR that the backend
hook can still wrap and execute end-to-end. The Phase-5 lowered-runtime
will swap the eager stub for a real kernel call; this pytest locks the
pipeline contract so that swap doesn't regress.
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


def _run_optimize(mlir_text: str, *, pass_pipeline: str | None = None) -> str:
    """Run the ``forge-opt --action=optimize`` driver over ``mlir_text``.

    When ``pass_pipeline`` is ``None`` the driver's default
    ``canonicalize,shape-inference,fuse`` chain is used. With an
    explicit value the chain is overridden to that single pass (Tasks
    10 + 11 use this).
    """
    args = [
        "/home/mei/projects/forgestack/.venv/bin/python3",
        "/home/mei/projects/forgestack/compiler/tools/forge-opt/forge_opt.py",
        "-",
        "--action=optimize",
    ]
    if pass_pipeline is not None:
        args.append(f"--pass-pipeline={pass_pipeline}")
    proc = subprocess.run(args, input=mlir_text, capture_output=True, text=True, check=True)
    return proc.stdout


@pytest.mark.parametrize("B,S,K,N", SHAPES)
def test_canonicalize_preserves_numerics(B: int, S: int, K: int, N: int) -> None:
    """Differential: post-canonicalize MLIR validates against eager for all 3 shapes.

    Both paths — the original ``compile()`` call and the
    post-canonicalize ``CompiledFusedMLP`` wrapper around the optimized
    MLIR — must produce output that :func:`validate` accepts against the
    eager reference. The canonicalize pass must therefore leave the IR
    observably equivalent (same shape / same fp32 numerics).
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

    # Path 2 — canonicalize-pass output, fed back through the same backend hook.
    optimized_mlir = _run_optimize(compiled.mlir_text, pass_pipeline="canonicalize")
    optimized_compiled = CompiledFusedMLP(model, [x, w, b], optimized_mlir)
    optimized_out = optimized_compiled(x, w, b)
    result_optimized = validate(optimized_out, eager_out)
    assert result_optimized.passed, (
        f"post-canonicalize MLIR failed validation: {result_optimized}\n"
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
