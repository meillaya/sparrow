"""Compiler lowering registry.

``capabilities.py`` describes what a backend is built to run.
This module describes what the compiler can actually emit today. The
partitioner needs both: a backend that lists an op but has no lowering
must not be selected.

``cpu`` is the only general lowering: the plan builder maps every op in
:data:`forge_torch.ops.CPU_ENTRY_POINTS` to a kernel. Triton has one
fused-group lowering (``gemm + bias + gelu``) emitted by the plan
builder's ``fuse_triton`` selection. Astra and MicroGPU remain empty, so
the partitioner fails closed for them.
"""

from __future__ import annotations

from typing import Final

from forge_torch.ops import CPU_ENTRY_POINTS

FUSED_TRITON_OPS: Final[frozenset[str]] = frozenset({"forge.tensor.fused_epilogue"})

COMPILER_LOWERINGS: Final[dict[str, frozenset[str]]] = {
    "cpu": frozenset(CPU_ENTRY_POINTS),
    "triton": FUSED_TRITON_OPS,
    "astra": frozenset(),
    "microgpu": frozenset(),
}

__all__ = ("COMPILER_LOWERINGS",)
