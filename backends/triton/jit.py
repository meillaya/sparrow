"""Triton JIT compile + launch wrapper (thin shim over the kernel module).

The shipped scope provides the public launch entry point used by the
Triton provider (Task 11) and a trivial `JITLauncher.compile` shim that
forwards to `triton.compile`. Per spec §7 the kernel artifact surface
(source + compile-time constants + launch grid + argument mapping +
shape guards + workspace) is owned by this module; the shipped kernel
emits the canonical template via the codegen stub.

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
    """Thin wrapper around ``triton.compile`` for the shipped kernel."""

    @staticmethod
    def compile(kernel_src: str, signature: str, *, target: str = "cuda"):
        """Compile a Triton kernel source string.

        The ``signature`` parameter is accepted for API forward-compatibility
        but is not used by the shipped shim; only ``kernel_src`` and
        ``target`` are forwarded to ``triton.compile``. Per spec §20 we
        never expose the Triton internal IR — only the compiled-kernel
        handle is returned.
        """
        del signature  # accepted for forward-compat; not used today
        import triton

        return triton.compile(kernel_src, target=target)
