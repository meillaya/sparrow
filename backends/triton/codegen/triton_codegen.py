"""emit_triton — return the canonical Triton source for the shipped kernel.

The shipped scope is a stub: it returns the canonical kernel source from
fused_gemm_bias_gelu.py so the codegen + JIT path is exercised end to end
without per-op kernel synthesis.

Reference: spec §7 Path A.
"""

from __future__ import annotations

from pathlib import Path

KERNEL_PATH = Path(__file__).parent.parent / "kernels" / "fused_gemm_bias_gelu.py"


def emit_triton(region: str) -> str:
    """Return the canonical fused_gemm_bias_gelu kernel source.

    The ``region`` argument is accepted for API forward-compatibility
    but is ignored today — the shipped codegen only emits the canonical
    template.
    """
    del region  # accepted for forward-compat; not used today
    return KERNEL_PATH.read_text()
