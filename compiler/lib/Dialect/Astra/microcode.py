"""Astra's fixed-width microcode instruction word."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class OpCode(IntEnum):
    """Operation selectors defined by the Astra microcode ISA."""

    NOP = 0x00
    DMA_LOAD = 0x01
    DMA_STORE = 0x02
    MATMUL = 0x03
    VECTOR_GELU = 0x04
    HALT = 0xFF


@dataclass(frozen=True, slots=True)
class MicrocodeInstr:
    """Decoded fields of one 64-bit Astra instruction."""

    opcode: OpCode
    reg_a: int
    reg_b: int
    imm: int
    flags: int


_WORD_MASK = 0xFFFF
_FLAG_MARKER = 0x8000
_PAYLOAD_MASK = _FLAG_MARKER - 1


def _check_word_field(value: int, name: str) -> None:
    if not 0 <= value <= _WORD_MASK:
        raise ValueError(f"{name} must fit in 16 bits, got {value}")


def encode(
    opcode: OpCode,
    reg_a: int,
    reg_b: int,
    imm: int = 0,
    flags: int = 0,
) -> int:
    """Pack an instruction into ``opcode|reg_a|reg_b|flags/imm`` fields.

    Each field is 16 bits, in descending significance order.  The final
    field is intentionally shared by an immediate and flags: an instruction
    carries either a 15-bit immediate or 15-bit flags, never both.  The high
    bit marks the flags form, leaving values such as the 8192-byte DMA
    transfer in the immediate form representable.
    """
    _check_word_field(reg_a, "reg_a")
    _check_word_field(reg_b, "reg_b")
    _check_word_field(imm, "imm")
    _check_word_field(flags, "flags")
    if imm and flags:
        raise ValueError("imm and flags share one 16-bit field; provide only one")
    if imm > _PAYLOAD_MASK:
        raise ValueError("imm must fit in the 15-bit immediate form")
    if flags > _PAYLOAD_MASK:
        raise ValueError("flags must fit in the 15-bit flags form")

    payload = imm if imm else (_FLAG_MARKER | flags if flags else 0)
    return (int(opcode) << 48) | (reg_a << 32) | (reg_b << 16) | payload


def decode(word: int) -> MicrocodeInstr:
    """Decode one 64-bit instruction word into its typed fields."""
    if not 0 <= word < (1 << 64):
        raise ValueError(f"instruction word must fit in 64 bits, got {word}")

    opcode_field = (word >> 48) & _WORD_MASK
    try:
        opcode = OpCode(opcode_field)
    except ValueError as exc:
        raise ValueError(f"unknown Astra opcode 0x{opcode_field:04x}") from exc

    payload = word & _WORD_MASK
    if payload & _FLAG_MARKER:
        imm = 0
        flags = payload & _PAYLOAD_MASK
    else:
        imm = payload
        flags = 0
    return MicrocodeInstr(
        opcode=opcode,
        reg_a=(word >> 32) & _WORD_MASK,
        reg_b=(word >> 16) & _WORD_MASK,
        imm=imm,
        flags=flags,
    )
