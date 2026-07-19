"""Triton JIT compile + launch wrapper (Phase 1: thin shim over the kernel module).

Phase 1 provides the public launch entry point used by the Triton provider
(Task 11) and a trivial `JITLauncher.compile` shim that forwards to
`triton.compile`. Phase 5 will:

* cache compiled kernels by signature,
* invalidate the cache when the codegen changes,
* expose the kernel artifact (source + compile-time constants + launch grid +
  argument mapping + shape guards + workspace) per spec §7.

Spec reference: .omo/plans/forgestack.md §7, §20.
"""

from __future__ import annotations

import torch

from backends.triton.kernels.fused_gemm_bias_gelu import (
    fused_gemm_bias_gelu as _fused_gemm_bias_gelu_impl,
)

__all__ = ["JITLauncher", "compile", "launch_fused_gemm_bias_gelu"]


def launch_fused_gemm_bias_gelu(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Public entry point used by the Triton provider (Task 11).

    Delegates to the kernel module so the launch semantics live with the
    source definition; the JIT launcher only owns compile + cache concerns.
    """
    return _fused_gemm_bias_gelu_impl(x, w, b)


class JITLauncher:
    """Phase 1: trivial wrapper. Phase 5 caches compiled kernels by signature."""

    @staticmethod
    def compile(kernel_src: str, signature: str, *, target: str = "cuda"):
        """Compile a Triton kernel source string.

        `signature` is reserved for Phase 5; Phase 1 only forwards `kernel_src`
        and `target` to `triton.compile`. Per spec §20 we never expose the
        Triton internal IR — only the compiled-kernel handle is returned.
        """
        del signature  # unused in Phase 1
        import triton

        return triton.compile(kernel_src, target=target)
