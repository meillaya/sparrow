"""MicroGPU's fixed-width SIMT bytecode instruction word (spec §11).

The microgpu backend exposes a single-warp SIMT bytecode ISA: each
instruction is exactly 16 bits wide, packed as ``opcode | dst | src |
imm`` (4 bits each). The 4-bit field widths match the spec's
single-warp model: 16 distinct opcodes (only 7 are wired today) and
16 distinct register slots (r0..r15 in the wired subset, but the
encode/decode layer accepts the full 4-bit range for forward-compat).

The seven shipped opcodes occupy the low half of the opcode space
(``RETURN=0x00`` through ``SIGMOID=0x05``). ``NOP=0xFF`` (the spec's
"all-ones" no-instruction convention) is encoded on the wire as
``0x0F`` — the highest 4-bit value, so the all-ones nibble pattern
is preserved at the wire level. The :class:`OpCode` enum keeps the
spec's literal ``0xFF`` value so any external test that imports the
enum sees the canonical number; encode/decode translate through
:func:`_wire_opcode` / :func:`_enum_opcode` automatically.

Reference: spec §11. Mirrors :mod:`compiler.lib.Dialect.Astra.microcode`
in spirit (typed opcode enum + frozen dataclass + 16-bit encode/decode
helpers) but uses a smaller 16-bit word with 4-bit fields instead of
the astra chip's 64-bit word.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


class OpCode(IntEnum):
    """Operation selectors defined by the MicroGPU SIMT bytecode ISA.

    The numeric values mirror spec §11. Six opcodes occupy the low
    half of the 4-bit opcode space; ``NOP`` carries the spec's literal
    ``0xFF`` value (the "all-ones" no-instruction sentinel) and is
    encoded on the wire as ``0x0F`` so it fits the 4-bit wire field.

    * ``LOAD_GLOBAL  (0x01)`` — read one 32-bit word from ``mem[imm]``
      into ``dst``. ``src`` is unused.
    * ``STORE_GLOBAL (0x02)`` — write ``src`` to ``mem[imm]``. ``dst``
      is unused.
    * ``ADD (0x03)``            — ``dst = dst + src`` (register-register).
    * ``MUL (0x04)``            — ``dst = dst * src`` (register-register).
    * ``SIGMOID (0x05)``        — ``dst = sigmoid(dst)``.
    * ``RETURN (0x00)``         — terminate the warp; subsequent
      instructions are not fetched.
    * ``NOP (0xFF)``            — no-op (encoded on the wire as ``0x0F``).
    """

    RETURN = 0x00
    LOAD_GLOBAL = 0x01
    STORE_GLOBAL = 0x02
    ADD = 0x03
    MUL = 0x04
    SIGMOID = 0x05
    NOP = 0xFF


#: Public per-instruction cycle counts consumed by the cost model.
#: LOAD/STORE move one i32 word, arithmetic is one cycle, and SIGMOID
#: is the reference transcendental lookup cost. RETURN terminates the
#: warp and adds no cycles beyond the dispatch already counted.
CYCLE_TABLE: Final[dict[OpCode, int]] = {
    OpCode.LOAD_GLOBAL: 4,
    OpCode.STORE_GLOBAL: 4,
    OpCode.ADD: 1,
    OpCode.MUL: 1,
    OpCode.SIGMOID: 8,
    OpCode.RETURN: 0,
}


@dataclass(frozen=True, slots=True)
class MicroOp:
    """Decoded fields of one 16-bit MicroGPU instruction.

    All three field-bearing values (``dst``, ``src``, ``imm``) are
    4-bit unsigned integers in the range ``[0, 15]``. ``opcode`` is
    the typed selector from :class:`OpCode`; the assembler/interpreter
    decide whether ``imm`` is interpreted as an address, a count, or
    as padding.
    """

    opcode: OpCode
    dst: int
    src: int
    imm: int


# 4-bit field masks (0xF) and shifts (0/4/8/12) keep the encode/decode
# helpers branchless and obvious. ``_FIELD_MASK = 0xF`` is the per-field
# bit-width; ``_OP_SHIFT = 12``, ``_DST_SHIFT = 8``, ``_SRC_SHIFT = 4``
# place each field in its respective nibble.
_FIELD_MASK: int = 0xF
_OP_SHIFT: int = 12
_DST_SHIFT: int = 8
_SRC_SHIFT: int = 4

# The spec lists ``NOP = 0xFF`` as a literal IntEnum value (the
# "all-ones" no-instruction sentinel). The wire opcode field is only
# 4 bits wide, so the encoder maps ``0xFF`` to the all-ones 4-bit value
# ``0x0F`` and the decoder reverses the mapping. This preserves both
# the spec's enum value and the 4-bit wire format.
_NOP_WIRE: int = 0x0F
_NOP_ENUM: int = 0xFF


def _wire_opcode(opcode: OpCode) -> int:
    """Translate the spec's :class:`OpCode` enum value to its 4-bit wire form."""
    raw = int(opcode)
    if raw == _NOP_ENUM:
        return _NOP_WIRE
    return raw & _FIELD_MASK


def _enum_opcode(wire: int) -> OpCode:
    """Translate a 4-bit wire opcode back to the spec's :class:`OpCode` enum value."""
    if wire == _NOP_WIRE:
        return OpCode.NOP
    return OpCode(wire)


def _check_field(value: int, name: str) -> None:
    """Reject values that don't fit in a 4-bit unsigned nibble."""
    if not 0 <= value <= _FIELD_MASK:
        raise ValueError(f"{name} must fit in 4 bits (0..15), got {value}")


def encode(opcode: OpCode, dst: int, src: int, imm: int = 0) -> int:
    """Pack an instruction into the 4-4-4-4 nibble layout.

    The result is a single 16-bit word: the high nibble is ``opcode``,
    followed by ``dst``, then ``src``, then ``imm`` in the low nibble.
    The :class:`OpCode` value is translated through
    :func:`_wire_opcode` so ``NOP`` (``0xFF``) emits the all-ones
    wire-nibble ``0x0F``. ``dst``, ``src``, and ``imm`` are validated
    against the 4-bit range; out-of-range values raise
    :class:`ValueError` rather than silently truncating.
    """
    wire_op = _wire_opcode(opcode)
    _check_field(wire_op, "opcode")
    _check_field(dst, "dst")
    _check_field(src, "src")
    _check_field(imm, "imm")
    return (wire_op << _OP_SHIFT) | (dst << _DST_SHIFT) | (src << _SRC_SHIFT) | imm


def decode(word: int) -> MicroOp:
    """Decode one 16-bit instruction word into its typed fields.

    Raises :class:`ValueError` if ``word`` doesn't fit in 16 bits, or if
    the high nibble does not match a known :class:`OpCode`.
    """
    if not 0 <= word < (1 << 16):
        raise ValueError(f"instruction word must fit in 16 bits, got {word}")

    wire_op = (word >> _OP_SHIFT) & _FIELD_MASK
    try:
        opcode = _enum_opcode(wire_op)
    except ValueError as exc:
        raise ValueError(f"unknown MicroGPU wire opcode 0x{wire_op:01x}") from exc

    return MicroOp(
        opcode=opcode,
        dst=(word >> _DST_SHIFT) & _FIELD_MASK,
        src=(word >> _SRC_SHIFT) & _FIELD_MASK,
        imm=word & _FIELD_MASK,
    )


__all__ = ("MicroOp", "OpCode", "decode", "encode")
