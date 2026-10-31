"""Sparrow Triton kernel template tests (Phase 1: shape + parse + warmup).

These tests validate the *kernel artifact surface* (source, AST shape,
constexpr signature, codegen stub) without requiring a CUDA device. The
numerical-correctness test on the last case *does* need CUDA; on a CPU-only
host it is skipped with a clear message rather than being silently dropped.

Reference: phase 1 plan §7 Path A.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

pytest.importorskip("triton", reason="Triton backend tests require the optional triton package")

from backends.triton.codegen.triton_codegen import emit_triton
from backends.triton.jit import JITLauncher, launch_fused_gemm_bias_gelu
from backends.triton.kernels.fused_gemm_bias_gelu import (
    _fused_gemm_bias_gelu_kernel,
    fused_gemm_bias_gelu,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kernel_src() -> str:
    """Return the canonical source string for the kernel.

    `inspect.getsource` does not work on `JITFunction` (it raises
    `TypeError: ... got JITFunction`), so we read the source via the
    `JITFunction.src` attribute that Triton stores at decoration time.
    """
    src = _fused_gemm_bias_gelu_kernel.src
    assert isinstance(src, str) and src.strip(), (
        f"JITFunction.src returned non-string: {type(src).__name__}"
    )
    return src


def _kernel_ast() -> ast.Module:
    """Parse the kernel source into an AST module for structural assertions.

    `kernel.parse()` works even on a CPU-only host (no driver needed) because
    it just runs `ast.parse` on the stored source.
    """
    parsed = _fused_gemm_bias_gelu_kernel.parse()
    assert isinstance(parsed, ast.Module)
    return parsed


def _signature_defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


# ---------------------------------------------------------------------------
# 1. Source-level validation
# ---------------------------------------------------------------------------


def test_kernel_source_parses() -> None:
    """The kernel source must be well-formed and contain GELU + GEMM primitives."""
    src = _kernel_src()

    # Well-formed Python: AST-parses without raising.
    ast.parse(src)

    # It's the fused-GEMM-GELU variant: must mention `tl.dot` and `tl.erf`.
    # (Bias addition lives next to `tl.erf`, so a positive erf match
    # guarantees the GELU branch was reached.)
    assert "tl.dot" in src, "kernel missing tl.dot (GEMM primitive)"
    assert "tl.erf" in src, "kernel missing tl.erf (GELU primitive)"
    assert "tl.load" in src, "kernel missing tl.load"
    assert "tl.store" in src, "kernel missing tl.store"


# ---------------------------------------------------------------------------
# 2. Constexpr signature surface
# ---------------------------------------------------------------------------


def test_kernel_metadata() -> None:
    """Kernel exposes BLOCK_M, BLOCK_N, BLOCK_K constexpr params (spec §7)."""
    parsed = _kernel_ast()
    assert parsed.body, "kernel AST has no top-level statements"
    func_def = parsed.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    arg_names = [a.arg for a in func_def.args.args]

    # Block-size knobs are the spec §7 named tile params.
    for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        assert name in arg_names, f"kernel missing constexpr param {name}"

    # The kernel must carry a `tl.constexpr` annotation on the block sizes;
    # otherwise they cannot be specialized at launch time.
    annotated = {a.arg: a.annotation for a in func_def.args.args}
    for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        ann = annotated[name]
        assert ann is not None, f"{name} has no annotation"
        assert isinstance(ann, ast.Attribute) and ann.attr == "constexpr", (
            f"{name} is not declared `tl.constexpr`: {ast.dump(ann)}"
        )

    # The codegen wrapper must accept the same defaults the spec §7 example uses.
    sig = _signature_defaults(fused_gemm_bias_gelu)
    assert sig["block_m"] == 64
    assert sig["block_n"] == 128
    assert sig["block_k"] == 32
    assert sig["num_warps"] == 8
    assert sig["num_stages"] == 4


# ---------------------------------------------------------------------------
# 3. Codegen stub returns the canonical kernel source
# ---------------------------------------------------------------------------


def test_codegen_returns_canonical_source() -> None:
    """Phase-1 codegen is a STUB: returns the canonical kernel source unchanged."""
    src = emit_triton(region="ignored")
    canonical = _kernel_src()
    assert canonical in src, (
        "emit_triton() must return the canonical kernel source from "
        "backends/triton/kernels/fused_gemm_bias_gelu.py"
    )
    # The `region` argument is documented as Phase-5 forward-compat; Phase 1
    # ignores it. Confirm by calling again with a different region value.
    src2 = emit_triton(region="forge.tensor.fused_epilogue")
    assert src2 == src, "emit_triton() must ignore the region argument in Phase 1"


# ---------------------------------------------------------------------------
# 4. Numerical equivalence — CUDA-only, skipped on CPU-only hosts
# ---------------------------------------------------------------------------


def test_fused_gemm_bias_gelu_shape_cpu_reference() -> None:
    """If CUDA is available: launch and compare against `torch.nn.functional.gelu`.

    On a CPU-only host (the common dev environment for Sparrow Phase 1)
    the Triton driver is unavailable — `kernel.warmup` and `triton.compile`
    both raise `RuntimeError: 0 active drivers`. The kernel source itself is
    still validated above via AST + `JITFunction.src`; this test only adds
    numerical correctness when the driver is present.
    """
    if not torch.cuda.is_available():
        pytest.skip("Triton kernel requires CUDA; CPU fallback implemented in Task 11")

    torch.manual_seed(0)
    M, K, N = 128, 64, 128
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")
    w = torch.randn(K, N, dtype=torch.float32, device="cuda")
    b = torch.randn(N, dtype=torch.float32, device="cuda")

    y = fused_gemm_bias_gelu(x, w, b)
    assert y.shape == (M, N), f"expected shape {(M, N)}, got {tuple(y.shape)}"

    expected = torch.nn.functional.gelu(x @ w + b, approximate="none")
    torch.testing.assert_close(y, expected, atol=1e-3, rtol=1e-3)

    # Also exercise the JIT-launch wrapper path to confirm it delegates
    # correctly to the kernel module (no driver-level surprises).
    y_via_wrapper = launch_fused_gemm_bias_gelu(x, w, b)
    torch.testing.assert_close(y_via_wrapper, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 5. JITLauncher.compile shim
# ---------------------------------------------------------------------------


def test_jit_launcher_compile_signature() -> None:
    """Phase-1 JITLauncher.compile is a thin shim over triton.compile.

    We don't call .compile() (would need an active driver); just verify the
    static method exists with the expected signature so callers can wire up.
    """
    import inspect

    params = inspect.signature(JITLauncher.compile).parameters
    assert "kernel_src" in params
    assert "signature" in params
    assert "target" in params
    assert params["target"].default == "cuda"
