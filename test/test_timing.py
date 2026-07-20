"""Tests for the Astra timing simulator."""

from __future__ import annotations

import pytest

from backends.astra.sim.functional import DRAM_BYTES, TILE_BYTES, TILE_DIM
from backends.astra.sim.timing import SimMetrics, run_timing
from compiler.lib.Dialect.Astra.assembler import assemble


def test_timing_simulator_populates_metrics() -> None:
    """Given a mixed program, timing mode reports the cycle-table totals."""
    k = 2
    operand_bytes = TILE_DIM * k * 4
    program = f"""\
DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes={operand_bytes}
DMA_LOAD dst=SRAM[0x{operand_bytes:x}], src=DRAM[0x{operand_bytes:x}], bytes={operand_bytes}
MATMUL a=SRAM[0x0], b=SRAM[0x{operand_bytes:x}], k={k}
VECTOR_GELU dst=SRAM[0x{2 * TILE_BYTES:x}], src=SRAM[0x0], elements={TILE_DIM * TILE_DIM}
DMA_STORE dst=DRAM[0x{2 * operand_bytes:x}], src=SRAM[0x{2 * TILE_BYTES:x}], bytes={TILE_BYTES}
HALT
"""

    _, metrics = run_timing(assemble(program), dram=bytearray(DRAM_BYTES))

    assert isinstance(metrics, SimMetrics)
    assert metrics.total_cycles == 33
    assert metrics.matrix_array_utilization == pytest.approx(20 / 33)
    assert 0.0 <= metrics.scratchpad_utilization <= 1.0
