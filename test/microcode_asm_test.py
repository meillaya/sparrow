"""Text assembler/disassembler tests for Astra microcode."""

from __future__ import annotations

import pytest

from compiler.lib.Dialect.Astra.assembler import assemble, disassemble
from compiler.lib.Dialect.Astra.microcode import OpCode, encode

TEXT_CASES = [
    (
        "DMA_LOAD  dst=SRAM[0x100], src=DRAM[0x2000], bytes=8192",
        encode(OpCode.DMA_LOAD, 0x100, 0x2000, imm=8192),
    ),
    (
        "DMA_STORE dst=DRAM[0x3000], src=SRAM[0x300], bytes=4096",
        encode(OpCode.DMA_STORE, 0x3000, 0x300, imm=4096),
    ),
    (
        "MATMUL a=SRAM[0x100], b=SRAM[0x200], k=16",
        encode(OpCode.MATMUL, 0x100, 0x200, imm=16),
    ),
    (
        "VECTOR_GELU dst=SRAM[0x300], src=SRAM[0x300], elements=1024",
        encode(OpCode.VECTOR_GELU, 0x300, 0x300, imm=1024),
    ),
    ("NOP", encode(OpCode.NOP, 0, 0)),
    ("HALT", encode(OpCode.HALT, 0, 0)),
]


@pytest.mark.parametrize(("text", "expected"), TEXT_CASES)
def test_assemble_spec_operation(text: str, expected: int) -> None:
    """Given one textual operation, assembly produces its instruction word."""
    assert assemble(text) == [expected]


@pytest.mark.parametrize(("text", "expected"), TEXT_CASES)
def test_disassemble_round_trip(text: str, expected: int) -> None:
    """Given one instruction word, disassembly reassembles to the same word."""
    words = assemble(text)

    assert disassemble(words) == disassemble([expected])
    assert assemble(disassemble(words)) == words


def test_assemble_ignores_blank_lines_and_comments() -> None:
    """Given formatting-only lines, assembly emits only executable operations."""
    text = "\n# DMA staging\nNOP\n\nHALT\n"

    assert assemble(text) == [encode(OpCode.NOP, 0, 0), encode(OpCode.HALT, 0, 0)]


def test_assemble_rejects_malformed_lines() -> None:
    """Given an unknown or incomplete line, assembly reports its line number."""
    with pytest.raises(ValueError, match=r"line 1.*missing bytes, src"):
        assemble("DMA_LOAD dst=SRAM[0x100]")
