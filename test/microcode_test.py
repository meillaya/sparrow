"""Round-trip coverage for the Astra microcode instruction word."""

from __future__ import annotations

import pytest

from compiler.lib.Dialect.Astra.microcode import MicrocodeInstr, OpCode, decode, encode


@pytest.mark.parametrize(
    ("opcode", "reg_a", "reg_b", "imm"),
    [
        (OpCode.NOP, 0, 0, 0),
        (OpCode.DMA_LOAD, 0x100, 0x2000, 8192),
        (OpCode.DMA_STORE, 0x300, 0x4000, 4096),
        (OpCode.MATMUL, 16, 16, 16),
        (OpCode.VECTOR_GELU, 0x300, 0x300, 0),
        (OpCode.HALT, 0, 0, 0),
    ],
)
def test_encode_decode_round_trip(opcode: OpCode, reg_a: int, reg_b: int, imm: int) -> None:
    """Given one ISA opcode, encoding and decoding preserves its fields."""
    word = encode(opcode, reg_a, reg_b, imm=imm)

    decoded = decode(word)

    assert decoded == MicrocodeInstr(opcode, reg_a, reg_b, imm, 0)
    assert 0 <= word < (1 << 64)


def test_flags_use_the_shared_final_word_field() -> None:
    """Given a flag-only instruction, the shared 16-bit field round-trips flags."""
    word = encode(OpCode.NOP, 0, 0, flags=0x12)

    assert decode(word) == MicrocodeInstr(OpCode.NOP, 0, 0, 0, 0x12)


def test_encode_rejects_simultaneous_imm_and_flags() -> None:
    """Given both alternatives, encoding rejects an ambiguous final field."""
    with pytest.raises(ValueError, match="imm and flags"):
        _ = encode(OpCode.MATMUL, 1, 2, imm=3, flags=4)
