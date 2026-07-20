"""Cycle accounting for the Astra functional simulator."""

# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, assert_never

from backends.astra.sim.functional import FunctionalResult, run_functional
from compiler.lib.Dialect.Astra.microcode import OpCode, decode

_CYCLE_TABLE: Final[dict[OpCode, int]] = {
    OpCode.DMA_LOAD: 4,
    OpCode.DMA_STORE: 4,
    OpCode.MATMUL: 16,
    OpCode.VECTOR_GELU: 4,
    OpCode.NOP: 1,
    OpCode.HALT: 1,
}


@dataclass(frozen=True, slots=True)
class SimMetrics:
    """Timing and resource metrics for one simulation."""

    total_cycles: int
    dma_stalled_cycles: int
    compute_stalled_cycles: int
    matrix_array_utilization: float
    scratchpad_utilization: float
    bytes_transferred: int


def run_timing(
    words: Sequence[int],
    *,
    dram: bytearray | None = None,
    sram: bytearray | None = None,
) -> tuple[FunctionalResult, SimMetrics]:
    """Execute microcode and attach sequential per-instruction timing."""
    result = run_functional(words, dram=dram, sram=sram)
    total_cycles = 0
    active_compute_cycles = 0

    for word in words:
        instruction = decode(word)
        cycles = _CYCLE_TABLE[instruction.opcode]
        total_cycles += cycles
        match instruction.opcode:
            case OpCode.MATMUL | OpCode.VECTOR_GELU:
                active_compute_cycles += cycles
            case OpCode.DMA_LOAD | OpCode.DMA_STORE | OpCode.NOP:
                pass
            case OpCode.HALT:
                break
            case unreachable:
                assert_never(unreachable)

    utilization = active_compute_cycles / total_cycles if total_cycles else 0.0
    return result, SimMetrics(
        total_cycles=total_cycles,
        dma_stalled_cycles=0,
        compute_stalled_cycles=0,
        matrix_array_utilization=utilization,
        scratchpad_utilization=0.0,
        bytes_transferred=0,
    )
