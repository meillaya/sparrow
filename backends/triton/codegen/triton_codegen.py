"""emit_triton — generate Triton source from a forge.tensor region (Phase 5).

Phase 1 is a STUB: it returns the canonical kernel source from
fused_gemm_bias_gelu.py. Phase 5 will wire it up to actually read
forge.tensor.gemm + forge.tensor.fused_epilogue regions and emit per-op kernels.

Reference: spec §7 Path A.
"""

from __future__ import annotations

from pathlib import Path

KERNEL_PATH = Path(__file__).parent.parent / "kernels" / "fused_gemm_bias_gelu.py"


def emit_triton(region: str) -> str:
    """Phase 1: return the canonical fused_gemm_bias_gelu kernel source.

    The `region` argument is accepted for Phase-5 forward-compatibility but is
    ignored today — Phase 1 only emits the canonical template.
    """
    del region  # unused in Phase 1
    return KERNEL_PATH.read_text()
