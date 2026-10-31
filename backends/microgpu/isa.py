"""MicroGPU's 32-bit SIMT bytecode instruction word (spec §11).

Version 2 widens the word from 16 to 32 bits so immediates can address
more than 16 words. The layout is ``opcode:4 | dst:6 | src:6 | imm:16``
— 64 register slots (32 wired), 16 opcodes, and a 65,536-word address
space (256 KiB of fp32/i32 data).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


class OpCode(IntEnum):
    """Operation selectors defined by the MicroGPU SIMT bytecode ISA."""

    RETURN = 0x00
    LOAD_GLOBAL = 0x01
    STORE_GLOBAL = 0x02
    ADD = 0x03
    MUL = 0x04
    SIGMOID = 0x05
    MATMUL = 0x06
    FADD = 0x07
    FMUL = 0x08
    FRELU = 0x09
    NOP = 0xFF


#: Public per-instruction cycle counts consumed by the cost model.
CYCLE_TABLE: Final[dict[OpCode, int]] = {
    OpCode.LOAD_GLOBAL: 4,
    OpCode.STORE_GLOBAL: 4,
    OpCode.ADD: 1,
    OpCode.MUL: 1,
    OpCode.SIGMOID: 8,
    OpCode.MATMUL: 16,
    OpCode.FADD: 1,
    OpCode.FMUL: 1,
    OpCode.FRELU: 1,
    OpCode.RETURN: 0,
}


@dataclass(frozen=True, slots=True)
class MicroOp:
    """Decoded fields of one 32-bit MicroGPU instruction."""

    opcode: OpCode
    dst: int
    src: int
    imm: int


_FIELD_MASK: int = 0x3F
_OP_MASK: int = 0xF
_OP_SHIFT: int = 28
_DST_SHIFT: int = 22
_SRC_SHIFT: int = 16
_IMM_MASK: int = 0xFFFF

_NOP_WIRE: int = 0x0F
_NOP_ENUM: int = 0xFF


def _wire_opcode(opcode: OpCode) -> int:
    raw = int(opcode)
    if raw == _NOP_ENUM:
        return _NOP_WIRE
    return raw & _OP_MASK


def _enum_opcode(wire: int) -> OpCode:
    if wire == _NOP_WIRE:
        return OpCode.NOP
    return OpCode(wire)


def _check_field(value: int, name: str, maximum: int = _FIELD_MASK) -> None:
    if not 0 <= value <= maximum:
        raise ValueError(
            f"{name} must fit in {maximum.bit_length()} bits (0..{maximum}), got {value}"
        )


def encode(opcode: OpCode, dst: int, src: int, imm: int = 0) -> int:
    """Pack an instruction into the 4-6-6-16 layout."""
    wire_op = _wire_opcode(opcode)
    _check_field(wire_op, "opcode", _OP_MASK)
    _check_field(dst, "dst")
    _check_field(src, "src")
    _check_field(imm, "imm", _IMM_MASK)
    return (wire_op << _OP_SHIFT) | (dst << _DST_SHIFT) | (src << _SRC_SHIFT) | imm


def decode(word: int) -> MicroOp:
    """Decode one 32-bit instruction word into its typed fields."""
    if not 0 <= word < (1 << 32):
        raise ValueError(f"instruction word must fit in 32 bits, got {word}")

    wire_op = (word >> _OP_SHIFT) & _OP_MASK
    try:
        opcode = _enum_opcode(wire_op)
    except ValueError as exc:
        raise ValueError(f"unknown MicroGPU wire opcode 0x{wire_op:01x}") from exc

    return MicroOp(
        opcode=opcode,
        dst=(word >> _DST_SHIFT) & _FIELD_MASK,
        src=(word >> _SRC_SHIFT) & _FIELD_MASK,
        imm=word & _IMM_MASK,
    )


__all__ = ("CYCLE_TABLE", "MicroOp", "OpCode", "decode", "encode")
