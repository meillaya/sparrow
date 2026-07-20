"""Analytical cost model reading Phase 4 cost stubs + per-op cycle tables (Phase 7 Wave 2)."""

# frontend/python/forge_torch/cost_estimator.py
#
# Phase 7 analytical CostModel (spec §6, §20).
#
# The Phase 7 partitioner (Task 18) consumes :func:`estimate_cost` to rank
# candidate backends for a region WITHOUT measuring — spec §20 forbids
# measured cost models at partition time, and a small table-driven estimate
# is what the spec asks for. Cross-region DMA costs for transitions to /
# from the Astra-sim scratchpad come from :func:`cross_region_cost` (used
# by the partitioner's boundary step).
#
# Sources (do not duplicate — reference, not redefine):
#   * per-op cycle tables — `frontend/python/forge_torch/_phase7_cycle_tables.py`
#   * per-backend capability registry — `frontend/python/forge_torch/capabilities.py`
#   * compilation-report cost stubs — `frontend/python/forge_torch/_phase4_stub_reader.py`
#     (Task 9; the partitioner multiplies these stub strings onto the
#     analytical estimate to fold in historical ratios).

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypedDict

from forge_torch._phase7_cycle_tables import (
    ASTRA_SIM_CYCLES,
    MICROGPU_SIM_CYCLES,
    TRITON_CPU_CYCLES,
)

# Phase 7 cost model is fp32-only today (matches every shipped backend's
# supported_types frozenset in capabilities.py).
_FP32_BYTES: Final = 4


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEstimate:
    """Triplet returned by :func:`estimate_cost`.

    All three fields are non-negative integers. ``bytes_transferred`` is
    non-zero only for Astra regions (DMA load / store bytes); Triton-CPU
    and MicroGPU-sim run in-place without a DMA model.
    """

    compute_cycles: int
    memory_bytes: int
    bytes_transferred: int


class RegionOp(TypedDict):
    """Schema for a single region item — dialect op name + operand shape.

    ``shape`` is the per-op operand shape (``[M, N, K]`` for matmul-style
    ops, ``[bytes]`` for Astra DMA ops, ``[M, N]`` for elementwise ops).
    """

    op: str
    shape: list[int]


# ---------------------------------------------------------------------------
# Dialect op → backend cycle-table op name.
#
# Keys mirror the ``forge.<dialect>.<op>`` spellings consumed by the
# partitioner input. A value of ``None`` means "this backend has no
# native cycle-table entry for this op" — the cost model falls back to
# the per-backend default path (e.g. MicroGPU-sim GELU is composed from
# two sigmoid invocations per spec §20 / plan §16).
# ---------------------------------------------------------------------------

_DIALECT_TO_BACKEND_OP: Final[dict[str, dict[str, str | None]]] = {
    # forge.tensor ops — Path A reference on Triton; Astra microcode ISA;
    # MicroGPU SIMT bytecode.
    "forge.tensor.gemm": {
        "triton": "matmul",
        "astra": "matmul",
        "microgpu": "add",
    },
    "forge.tensor.fused_epilogue": {
        "triton": "matmul",
        "astra": "matmul",
        "microgpu": "add",
    },
    "forge.tensor.broadcast_add": {
        "triton": "broadcast_add",
        "astra": None,
        "microgpu": "add",
    },
    "forge.tensor.gelu": {
        "triton": "gelu",
        "astra": "vector_gelu",
        "microgpu": None,  # composed — see _microgpu_cycles
    },
    "forge.tensor.layer_norm": {
        "triton": "layer_norm",
        "astra": None,
        "microgpu": None,
    },
    "forge.tensor.softmax": {
        "triton": "softmax",
        "astra": None,
        "microgpu": None,
    },
    "forge.tensor.scaled_dot_product_attention": {
        "triton": "matmul",
        "astra": None,
        "microgpu": "add",
    },
    "forge.tensor.reshape": {
        "triton": "matmul",
        "astra": None,
        "microgpu": None,
    },
    # forge.microcode ops — Phase 4 microcode ISA spelling.
    "forge.microcode.dma_load": {
        "triton": None,
        "astra": "dma_load",
        "microgpu": None,
    },
    "forge.microcode.dma_store": {
        "triton": None,
        "astra": "dma_store",
        "microgpu": None,
    },
    "forge.microcode.matmul": {
        "triton": "matmul",
        "astra": "matmul",
        "microgpu": "add",
    },
    "forge.microcode.vector_gelu": {
        "triton": "gelu",
        "astra": "vector_gelu",
        "microgpu": None,  # composed — see _microgpu_cycles
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_bytes(shape: Sequence[int]) -> int:
    """Total fp32 byte footprint of a tensor with ``prod(shape)`` elements."""
    product = 1
    for dim in shape:
        product *= dim
    return product * _FP32_BYTES


def _output_bytes(op: str, shape: Sequence[int]) -> int:
    """Bytes of the op's output tensor — the live bytes crossing a boundary.

    Matmul-style ops use the output rank (``M * N``); elementwise ops use
    ``prod(shape)``; DMA ops use ``shape[0]`` (the byte count the op moves).
    """
    if "dma_" in op:
        return int(shape[0])
    if "matmul" in op or "gemm" in op:
        return int(shape[0]) * int(shape[1]) * _FP32_BYTES
    return _tensor_bytes(shape)


def _peak_live_bytes(region: Sequence[dict]) -> int:
    """Peak fp32 bytes live in the region (sum of per-op tensor footprints)."""
    return sum(_tensor_bytes(item["shape"]) for item in region)


# ---------------------------------------------------------------------------
# Per-backend per-op cycle dispatchers.
# ---------------------------------------------------------------------------


def _triton_cycles(op: str, shape: Sequence[int]) -> int:
    """Cycle count for a Triton-CPU op. All entries take a ``shape`` tuple."""
    return TRITON_CPU_CYCLES[op](tuple(shape))


def _astra_cycles(op: str, shape: Sequence[int]) -> int:
    """Cycle count for an Astra-sim op.

    The Astra cycle table overloads argument types per spec §9:
    ``matmul`` takes the ``(M, N, K)`` tuple, ``vector_gelu`` takes the
    element count, and ``dma_load`` / ``dma_store`` take the byte count.
    The region dict's ``shape`` list dispatches accordingly.
    """
    fn = ASTRA_SIM_CYCLES[op]
    if op in {"dma_load", "dma_store"}:
        return fn(int(shape[0]))
    if op == "vector_gelu":
        return fn(int(shape[0]))
    return fn(tuple(shape))


def _astra_dma_bytes(op: str, shape: Sequence[int]) -> int:
    """Bytes transferred by an Astra DMA op (load or store)."""
    return int(shape[0]) if op in {"dma_load", "dma_store"} else 0


# Per plan §16 / spec §20: MicroGPU-sim GELU is approximated as two
# sigmoid invocations on the per-instruction cycle table
# (8 + 8 = 16 cycles flat). The shipped SIMT ISA has no native GELU
# instruction, so the partitioner routes ``forge.tensor.gelu`` regions
# to either Triton (single op) or Astra (``vector_gelu``); MicroGPU is
# only a cost target when the user explicitly opts in via ``target=``.
_MICROGPU_GELU_CYCLES: Final = MICROGPU_SIM_CYCLES["sigmoid"] + MICROGPU_SIM_CYCLES["sigmoid"]


def _microgpu_cycles(op: str | None, dialect_op: str) -> int:
    """Cycle count for a MicroGPU-sim op.

    The MicroGPU SIMT cycle table is flat per-instruction (no shape
    dependence). ``gelu`` has no native ISA entry, so the cost model
    composes it from two sigmoid operations as above.
    """
    if op is not None:
        return MICROGPU_SIM_CYCLES[op]
    if "gelu" in dialect_op:
        return _MICROGPU_GELU_CYCLES
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_cost(
    region: Sequence[dict],
    backend_name: str,
) -> CostEstimate:
    """Estimate the analytical cost of ``region`` on ``backend_name``.

    Each region item is a dict with at minimum ``{"op": str, "shape":
    list[int]}`` (see :class:`RegionOp`). ``compute_cycles`` is the sum
    of per-op cycle counts from the per-backend cycle tables;
    ``memory_bytes`` is the sum of per-op tensor footprints at fp32
    (the ``"peak bytes live"`` proxy per spec §6); ``bytes_transferred``
    is the DMA byte total for Astra regions (zero on Triton / MicroGPU).
    """
    compute = 0
    memory = 0
    xfer = 0
    for item in region:
        dialect_op = item["op"]
        shape = item["shape"]
        backend_op = _DIALECT_TO_BACKEND_OP.get(dialect_op, {}).get(backend_name)

        if backend_name == "triton":
            if backend_op is not None:
                compute += _triton_cycles(backend_op, shape)
        elif backend_name == "astra":
            if backend_op is not None:
                compute += _astra_cycles(backend_op, shape)
                xfer += _astra_dma_bytes(backend_op, shape)
        elif backend_name == "microgpu":
            compute += _microgpu_cycles(backend_op, dialect_op)
        # Unknown backend: silently skip. The partitioner's capability
        # check should have rejected it before reaching this estimator.

        memory += _tensor_bytes(shape)

    return CostEstimate(
        compute_cycles=compute,
        memory_bytes=memory,
        bytes_transferred=xfer,
    )


def cross_region_cost(
    current_region: Sequence[dict],
    current_backend: str,
    next_backend: str,
) -> int:
    """Bytes transferred across a region boundary.

    Non-Astra crossings cost zero (Triton-CPU and MicroGPU-sim have no
    DMA model). Crossings that involve Astra either move a tensor into
    or out of the scratchpad (DMA-in / DMA-out) or move data between
    two adjacent Astra regions (sum of the trailing current region's
    explicit DMA load / store bytes).
    """
    # No DMA unless at least one side is Astra.
    if "astra" not in (current_backend, next_backend):
        return 0

    if current_backend == "astra" and next_backend == "astra":
        # Two adjacent Astra regions: bytes flowing across = the explicit
        # DMA load + store bytes in the trailing current region.
        total = 0
        for item in current_region:
            backend_op = _DIALECT_TO_BACKEND_OP.get(item["op"], {}).get("astra")
            if backend_op is not None:
                total += _astra_dma_bytes(backend_op, item["shape"])
        return total

    # Astra ↔ non-Astra: the live output tensor bytes of the trailing op
    # physically cross the boundary.
    if current_region:
        last = current_region[-1]
        return _output_bytes(last["op"], last["shape"])
    return 0


__all__ = ("CostEstimate", "RegionOp", "cross_region_cost", "estimate_cost")
