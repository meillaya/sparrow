"""Per-backend per-op cycle constants (Phase 7 Wave 0)."""

# frontend/python/forge_torch/_phase7_cycle_tables.py
#
# Per-op per-backend cycle constants for the Phase 7 partitioner (spec §16,
# §20). The partitioner consumes these tables to rank candidate backends for a
# region WITHOUT measuring — spec §20 forbids measured cost models at this
# stage, and a small table-driven model is what the spec asks for.
#
# Sources (do not duplicate — reference, not redefine):
#   * Triton-CPU    — `backends/triton/cpu/fused_gemm_bias_gelu.cpp` naive
#                     triple loop → matmul cost = M*N*K cycles.
#   * Astra-sim     — `backends/astra/sim/timing.py::_CYCLE_TABLE` per-instruction
#                     cycle count (DMA_LOAD=4, DMA_STORE=4, MATMUL=16, VECTOR_GELU=4).
#                     Matmul is per-tile: 16 cycles * number of 16x16 tiles.
#   * MicroGPU-sim  — `backends/microgpu/interp.py` per-instruction cycles
#                     (LOAD_GLOBAL=4, STORE_GLOBAL=4, ADD=1, MUL=1, SIGMOID=8,
#                     RETURN=0).
#
# Three backends, three tables, one stub-reader for the Phase 4 cost strings
# the partitioner consumes to scale its analytical estimate by the historical
# ratio (alternative_astra_cost, alternative_fpga_cost).

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from forge_torch.report import CompilationReport

# ---------------------------------------------------------------------------
# Triton-CPU — naive triple-loop kernel (Path A reference).
#
# The shipped kernel (fused_gemm_bias_gelu.cpp) is a literal `for m, for n,
# for k` triple loop with bias add + GELU epilogue; the cycle count therefore
# equals the number of multiply-adds performed. Elementwise ops (add, gelu,
# layer_norm, softmax) each touch every output element exactly once. The
# partitioner passes the op's output shape; the lambda returns the cycle count.
# ---------------------------------------------------------------------------

TRITON_CPU_CYCLES: Final[dict[str, Callable[[Sequence[int]], int]]] = {
    "matmul": lambda shape: shape[0] * shape[1] * shape[2],  # M*N*K
    "broadcast_add": lambda shape: shape[0] * shape[1],  # output M*N
    "gelu": lambda shape: shape[0] * shape[1],  # elementwise
    "layer_norm": lambda shape: shape[0] * shape[1],  # elementwise (mean+var+norm)
    "softmax": lambda shape: shape[0] * shape[1],  # elementwise (exp+sum+div)
}

# ---------------------------------------------------------------------------
# Astra-sim — microcode ISA, per-instruction cycle table.
#
# Per `backends/astra/sim/timing.py::_CYCLE_TABLE`: DMA_LOAD=4, DMA_STORE=4,
# MATMUL=16 (per tile), VECTOR_GELU=4 per element. Matmul cost is the number
# of 16x16 tiles x 16 cycles; the cycle-accurate simulator counts each tile's
# systolic-array invocation as 16 cycles regardless of K. The lambda takes the
# raw M/N/K triple and returns the total cycles for the tile sweep.
# ---------------------------------------------------------------------------

_ASTRA_TILE_DIM: Final = 16


def _astra_tiles(dim: int) -> int:
    """Ceiling-divide ``dim`` by the 16x16 systolic-array tile dimension."""
    return (dim + _ASTRA_TILE_DIM - 1) // _ASTRA_TILE_DIM


ASTRA_SIM_CYCLES: Final[dict[str, Callable[..., int]]] = {
    "dma_load": lambda bytes_count: 4 * bytes_count,  # 4 cycles per byte
    "dma_store": lambda bytes_count: 4 * bytes_count,  # 4 cycles per byte
    "matmul": lambda shape: (
        _astra_tiles(shape[0]) * _astra_tiles(shape[1]) * _astra_tiles(shape[2]) * 16
    ),  # 16 cycles per 16x16 tile
    "vector_gelu": lambda size: 4 * size,  # 4 cycles per element
}

# ---------------------------------------------------------------------------
# MicroGPU-sim — SIMT bytecode interpreter, per-instruction cycles.
#
# Per `backends/microgpu/interp.py`: LOAD_GLOBAL=4, STORE_GLOBAL=4, ADD=1,
# MUL=1, SIGMOID=8, RETURN=0. Every op is a fixed cycle cost (no shape
# dependence — the interpreter is a scalar in-order reference, not a SIMT
# throughput model). RETURN terminates the program and therefore costs 0
# extra cycles beyond what was already dispatched.
# ---------------------------------------------------------------------------

MICROGPU_SIM_CYCLES: Final[dict[str, int]] = {
    "load_global": 4,
    "store_global": 4,
    "add": 1,
    "mul": 1,
    "sigmoid": 8,
    "return": 0,
}


# ---------------------------------------------------------------------------
# Phase 4 cost stub reader.
#
# Phase 4's `CompilationReport` carries the two ``alternative_*_cost`` strings
# (e.g. ``"1.31x"``, ``"2.04x including transfer"``) that record how much
# slower the alternative backends would have been vs. the selected target.
# The Phase 7 partitioner consumes those strings as a scaling factor on the
# analytical cycle estimate (Task 9 of the plan calls these out explicitly).
# ---------------------------------------------------------------------------


def read_phase4_stubs(report: CompilationReport) -> dict[str, str]:
    """Return the Phase 4 ``alternative_*_cost`` strings as a stub dict.

    The keys mirror the dataclass field names verbatim so callers can use the
    dict as a drop-in for the report attributes without losing information.
    An empty report yields empty strings (not a KeyError), so the stub reader
    is safe to call on a freshly-instantiated :class:`CompilationReport`.
    """
    return {
        "alternative_astra_cost": report.alternative_astra_cost,
        "alternative_fpga_cost": report.alternative_fpga_cost,
    }


__all__ = (
    "ASTRA_SIM_CYCLES",
    "MICROGPU_SIM_CYCLES",
    "TRITON_CPU_CYCLES",
    "read_phase4_stubs",
)
