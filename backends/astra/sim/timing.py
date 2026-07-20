"""Cycle accounting for the Astra functional simulator."""

# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, assert_never

from backends.astra.sim.functional import (
    SRAM_BYTES,
    TILE_BYTES,
    TILE_DIM,
    FunctionalResult,
    run_functional,
)
from compiler.lib.Dialect.Astra.microcode import OpCode, decode

_CYCLE_TABLE: Final[dict[OpCode, int]] = {
    OpCode.DMA_LOAD: 4,
    OpCode.DMA_STORE: 4,
    OpCode.MATMUL: 16,
    OpCode.VECTOR_GELU: 4,
    OpCode.NOP: 1,
    OpCode.HALT: 1,
}
_FLOAT_BYTES: Final = 4


@dataclass(frozen=True, slots=True)
class SimMetrics:
    """Timing and resource metrics for one simulation."""

    total_cycles: int
    dma_stalled_cycles: int
    compute_stalled_cycles: int
    matrix_array_utilization: float
    scratchpad_utilization: float
    bytes_transferred: int


class _ResidencyTracker:
    """Mutable byte-occupancy map used to measure peak SRAM residency."""

    __slots__ = ("_resident", "current_bytes", "peak_bytes")

    def __init__(self) -> None:
        self._resident = bytearray(SRAM_BYTES)
        self.current_bytes = 0
        self.peak_bytes = 0

    def write(self, offset: int, size: int) -> None:
        end = offset + size
        newly_resident = size - sum(self._resident[offset:end])
        self._resident[offset:end] = b"\x01" * size
        self.current_bytes += newly_resident
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def free(self, offset: int, size: int) -> None:
        end = offset + size
        released = sum(self._resident[offset:end])
        self._resident[offset:end] = b"\x00" * size
        self.current_bytes -= released


def run_timing(
    words: Sequence[int],
    *,
    dram: bytearray | None = None,
    sram: bytearray | None = None,
) -> tuple[FunctionalResult, SimMetrics]:
    """Execute microcode and attach sequential per-instruction timing."""
    result = run_functional(words, dram=dram, sram=sram)
    residency = _ResidencyTracker()
    total_cycles = 0
    active_compute_cycles = 0

    for word in words:
        instruction = decode(word)
        cycles = _CYCLE_TABLE[instruction.opcode]
        total_cycles += cycles
        match instruction.opcode:
            case OpCode.DMA_LOAD:
                residency.write(instruction.reg_a, instruction.imm)
            case OpCode.DMA_STORE:
                residency.free(instruction.reg_b, instruction.imm)
            case OpCode.MATMUL:
                active_compute_cycles += cycles
                operand_bytes = TILE_DIM * instruction.imm * _FLOAT_BYTES
                residency.free(instruction.reg_a, operand_bytes)
                residency.free(instruction.reg_b, operand_bytes)
                residency.write(instruction.reg_a, TILE_BYTES)
            case OpCode.VECTOR_GELU:
                active_compute_cycles += cycles
                vector_bytes = instruction.imm * _FLOAT_BYTES
                residency.free(instruction.reg_b, vector_bytes)
                residency.write(instruction.reg_a, vector_bytes)
            case OpCode.NOP:
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
        scratchpad_utilization=residency.peak_bytes / SRAM_BYTES,
        bytes_transferred=0,
    )
