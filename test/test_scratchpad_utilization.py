"""Tests for timing-mode scratchpad utilization."""

from __future__ import annotations

import pytest

from backends.astra.sim.functional import DRAM_BYTES, SRAM_BYTES, TILE_BYTES
from backends.astra.sim.timing import run_timing
from compiler.lib.Dialect.Astra.assembler import assemble


def test_scratchpad_utilization_matches_hand_computed_peak() -> None:
    """Two resident tiles produce a 2 KiB peak in the 512 KiB SRAM."""
    program = f"""\
DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes={TILE_BYTES}
DMA_LOAD dst=SRAM[0x{TILE_BYTES:x}], src=DRAM[0x{TILE_BYTES:x}], bytes={TILE_BYTES}
MATMUL a=SRAM[0x0], b=SRAM[0x{TILE_BYTES:x}], k=16
VECTOR_GELU dst=SRAM[0x{2 * TILE_BYTES:x}], src=SRAM[0x0], elements=256
DMA_STORE dst=DRAM[0x0], src=SRAM[0x{2 * TILE_BYTES:x}], bytes={TILE_BYTES}
HALT
"""

    _, metrics = run_timing(assemble(program), dram=bytearray(DRAM_BYTES))

    assert metrics.scratchpad_utilization == pytest.approx((2 * TILE_BYTES) / SRAM_BYTES)
