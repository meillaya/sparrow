"""Tests for the Astra timing simulator."""

# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

import random
from typing import Final, assert_never

import numpy as np
import pytest

import backends.astra.sim as astra_sim
from backends.astra.sim.functional import (
    DRAM_BYTES,
    TILE_BYTES,
    TILE_DIM,
    run_functional,
)
from backends.astra.sim.timing import SimMetrics, run_timing
from compiler.lib.Dialect.Astra.assembler import assemble
from compiler.lib.Dialect.Astra.microcode import OpCode, encode

_RANDOM_OPS: Final = (
    OpCode.DMA_LOAD,
    OpCode.DMA_STORE,
    OpCode.MATMUL,
    OpCode.VECTOR_GELU,
    OpCode.NOP,
)


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


def test_entry_point_cross_checks_functional_and_timing_outputs() -> None:
    """Given one program, both public modes return equivalent tensors."""
    program = f"""\
DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes={TILE_BYTES}
VECTOR_GELU dst=SRAM[0x{TILE_BYTES:x}], src=SRAM[0x0], elements={TILE_DIM * TILE_DIM}
DMA_STORE dst=DRAM[0x0], src=SRAM[0x{TILE_BYTES:x}], bytes={TILE_BYTES}
HALT
"""
    words = assemble(program)

    functional, functional_metrics = astra_sim.run_microcode(words)
    timing, timing_metrics = astra_sim.run_microcode(words, mode="timing", metrics=True)

    assert functional is not None
    assert timing is not None
    assert np.allclose(functional, timing, atol=1e-4, rtol=0.0)
    assert functional_metrics is None
    assert timing_metrics is not None


def _random_program(seed: int) -> list[int]:
    rng = random.Random(seed)
    words: list[int] = []
    for _ in range(7):
        opcode = rng.choice(_RANDOM_OPS)
        sram_a = rng.randrange(8) * TILE_BYTES
        sram_b = rng.randrange(8) * TILE_BYTES
        dram_offset = rng.randrange(16) * TILE_BYTES
        match opcode:
            case OpCode.DMA_LOAD:
                words.append(encode(opcode, sram_a, dram_offset, imm=TILE_BYTES))
            case OpCode.DMA_STORE:
                words.append(encode(opcode, dram_offset, sram_a, imm=TILE_BYTES))
            case OpCode.MATMUL:
                words.append(encode(opcode, sram_a, sram_b, imm=TILE_DIM))
            case OpCode.VECTOR_GELU:
                words.append(
                    encode(
                        opcode,
                        sram_a,
                        sram_b,
                        imm=TILE_DIM * TILE_DIM,
                    )
                )
            case OpCode.NOP:
                words.append(encode(opcode, 0, 0))
            case unreachable:
                assert_never(unreachable)
    words.extend(
        (
            encode(OpCode.DMA_STORE, 0, 0, imm=TILE_BYTES),
            encode(OpCode.HALT, 0, 0),
        )
    )
    return words


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_random_microcode_functional_matches_timing_bytes(seed: int) -> None:
    rng = random.Random(seed)
    dram = bytearray(DRAM_BYTES)
    dram[: 16 * TILE_BYTES] = rng.randbytes(16 * TILE_BYTES)
    words = _random_program(seed)

    functional = run_functional(words, dram=bytearray(dram))
    timing, _ = run_timing(words, dram=bytearray(dram))

    assert functional.dram_writes == timing.dram_writes
    assert functional.sram == timing.sram


def test_bytes_transferred_sums_dma_loads_and_stores() -> None:
    program = """\
DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes=32
DMA_LOAD dst=SRAM[0x20], src=DRAM[0x20], bytes=64
DMA_STORE dst=DRAM[0x100], src=SRAM[0x0], bytes=16
HALT
"""

    _, metrics = run_timing(assemble(program), dram=bytearray(DRAM_BYTES))

    assert metrics.bytes_transferred == 32 + 64 + 16
