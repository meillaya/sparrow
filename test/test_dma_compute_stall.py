"""Tests for DMA-versus-compute stall decomposition."""

from __future__ import annotations

import backends.astra.sim.timing as timing
from backends.astra.sim.functional import DRAM_BYTES, TILE_BYTES
from compiler.lib.Dialect.Astra.assembler import assemble


def test_dma_and_compute_stalls_match_hand_computed_cycles() -> None:
    """A mixed sequence attributes busy-engine cycles to the idle engine."""
    program = f"""\
DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes={TILE_BYTES}
DMA_LOAD dst=SRAM[0x{TILE_BYTES:x}], src=DRAM[0x{TILE_BYTES:x}], bytes={TILE_BYTES}
MATMUL a=SRAM[0x0], b=SRAM[0x{TILE_BYTES:x}], k=16
VECTOR_GELU dst=SRAM[0x{2 * TILE_BYTES:x}], src=SRAM[0x0], elements=256
DMA_STORE dst=DRAM[0x0], src=SRAM[0x{2 * TILE_BYTES:x}], bytes={TILE_BYTES}
NOP
HALT
"""

    _, metrics = timing.run_timing(assemble(program), dram=bytearray(DRAM_BYTES))

    assert metrics.total_cycles == 34
    assert metrics.compute_stalled_cycles == 12
    assert metrics.dma_stalled_cycles == 20
    assert tuple(timing.StallReason) == (
        timing.StallReason.NONE,
        timing.StallReason.DMA,
        timing.StallReason.COMPUTE,
        timing.StallReason.BOTH,
    )
