"""Text assembler and disassembler for the MicroGPU SIMT bytecode ISA (spec §11).

Text format:

    LOAD_GLOBAL  r0, [0x1000]   # mem[0x1000] -> r0
    STORE_GLOBAL r1, [0x2000]   # r1 -> mem[0x2000]
    ADD  r2, r3                 # r2 = r2 + r3
    MUL  r2, r3                 # r2 = r2 * r3
    SIGMOID r2                  # r2 = sigmoid(r2)
    RETURN                      # end-of-warp
    NOP                         # no-op

The assembler/disassembler pair is symmetric: a round-trip
``assemble(text) -> disassemble(words) -> assemble(...)`` preserves
the word stream modulo whitespace and comments. The dialect bridge in
:func:`assemble_from_microgpu_dialect` /
:func:`disassemble_to_microgpu_dialect` is added in Task 72; this
module's first commit only ships the text path.

Reference: spec §11. Mirrors
:mod:`compiler.lib.Dialect.Astra.assembler` in shape.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from backends.microgpu.isa import OpCode, decode, encode


# Register literals ("r0".."r15") parsed from the textual assembly.
# Out-of-range register numbers are rejected by ``_parse_register``.
_REGISTER_RE: Final[re.Pattern[str]] = re.compile(r"^r(?P<num>[0-9]+)$")

# Memory operand "[0xADDR]" or "[DEC]" parsed from the textual assembly.
_MEMORY_RE: Final[re.Pattern[str]] = re.compile(r"^\[(?P<addr>0[xX][0-9a-fA-F]+|[0-9]+)\]$")


def _line_error(line_number: int, message: str) -> ValueError:
    """Build a line-numbered :class:`ValueError` for assembly errors."""
    return ValueError(f"line {line_number}: {message}")


def _parse_register(token: str, line_number: int) -> int:
    """Parse a textual register reference (``r0``..``r15``) into a 4-bit slot."""
    match = _REGISTER_RE.fullmatch(token)
    if match is None:
        raise _line_error(line_number, f"expected register like r0..r15, got {token!r}")
    number = int(match.group("num"))
    if number < 0 or number > 0xF:
        raise _line_error(line_number, f"register {token!r} does not fit in 4 bits")
    return number


def _parse_address(token: str, line_number: int) -> int:
    """Parse a bracketed memory operand (``[0x1000]``) into an integer address.

    The wire-format immediate field is only 4 bits wide, so addresses
    are constrained to ``[0, 15]``. Larger programs should treat the
    immediate as a base-offset pair (out of scope for the shipped
    single-warp model).
    """
    match = _MEMORY_RE.fullmatch(token)
    if match is None:
        raise _line_error(line_number, f"expected bracketed address like [0x1000], got {token!r}")
    try:
        address = int(match.group("addr"), 0)
    except ValueError as exc:
        raise _line_error(line_number, f"address must be an integer, got {token!r}") from exc
    if address < 0 or address > 0xF:
        raise _line_error(line_number, f"address {token!r} does not fit in 4-bit immediate")
    return address


def _split_operands(raw: str, line_number: int) -> list[str]:
    """Split a comma-separated operand list, stripping whitespace."""
    if not raw.strip():
        return []
    parts = [part.strip() for part in raw.split(",")]
    if any(part == "" for part in parts):
        raise _line_error(line_number, "empty operand in comma-separated list")
    return parts


def _parse_instruction(line: str, line_number: int) -> int:
    """Parse one textual instruction into a 16-bit word."""
    opcode_name, separator, raw_operands = line.partition(" ")
    if not opcode_name or (not separator and raw_operands):
        raise _line_error(line_number, f"malformed instruction {line!r}")
    try:
        opcode = OpCode[opcode_name.upper()]
    except KeyError as exc:
        raise _line_error(line_number, f"unknown opcode {opcode_name!r}") from exc

    operands = _split_operands(raw_operands, line_number)

    match opcode:
        case OpCode.RETURN | OpCode.NOP:
            if operands:
                raise _line_error(
                    line_number,
                    f"{opcode.name} takes no operands, got {operands!r}",
                )
            return encode(opcode, 0, 0, imm=0)
        case OpCode.LOAD_GLOBAL:
            if len(operands) != 2:
                raise _line_error(
                    line_number,
                    f"LOAD_GLOBAL expects 2 operands (dst, [addr]), got {operands!r}",
                )
            dst = _parse_register(operands[0], line_number)
            address = _parse_address(operands[1], line_number)
            return encode(opcode, dst, 0, imm=address)
        case OpCode.STORE_GLOBAL:
            if len(operands) != 2:
                raise _line_error(
                    line_number,
                    f"STORE_GLOBAL expects 2 operands (src, [addr]), got {operands!r}",
                )
            src = _parse_register(operands[0], line_number)
            address = _parse_address(operands[1], line_number)
            return encode(opcode, 0, src, imm=address)
        case OpCode.ADD | OpCode.MUL:
            if len(operands) != 2:
                raise _line_error(
                    line_number,
                    f"{opcode.name} expects 2 operands (dst, src), got {operands!r}",
                )
            dst = _parse_register(operands[0], line_number)
            src = _parse_register(operands[1], line_number)
            return encode(opcode, dst, src, imm=0)
        case OpCode.SIGMOID:
            if len(operands) != 1:
                raise _line_error(
                    line_number,
                    f"SIGMOID expects 1 operand (dst), got {operands!r}",
                )
            dst = _parse_register(operands[0], line_number)
            return encode(opcode, dst, 0, imm=0)


def assemble(text: str) -> list[int]:
    """Assemble one textual MicroGPU instruction per non-empty input line.

    Blank lines and ``# ...`` comments are ignored. Each non-empty line
    is parsed by :func:`_parse_instruction` and the resulting 16-bit
    word is appended to the output list in source order.
    """
    words: list[int] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        words.append(_parse_instruction(line, line_number))
    return words


def _format_register(slot: int) -> str:
    """Render a 4-bit register slot as ``r0``..``r15``."""
    return f"r{slot}"


def _format_address(address: int) -> str:
    """Render an integer address as ``[0xN]`` for the disassembler."""
    return f"[0x{address:x}]"


def disassemble(words: Sequence[int]) -> str:
    """Render instruction words in the canonical MicroGPU text format.

    Each word becomes exactly one line. The output round-trips back
    through :func:`assemble` and re-emits the same word stream (modulo
    whitespace and comments).
    """
    lines: list[str] = []
    for word in words:
        instr = decode(word)
        match instr.opcode:
            case OpCode.RETURN | OpCode.NOP:
                lines.append(instr.opcode.name)
            case OpCode.LOAD_GLOBAL:
                lines.append(
                    f"LOAD_GLOBAL  {_format_register(instr.dst)}, {_format_address(instr.imm)}"
                )
            case OpCode.STORE_GLOBAL:
                lines.append(
                    f"STORE_GLOBAL {_format_register(instr.src)}, {_format_address(instr.imm)}"
                )
            case OpCode.ADD:
                lines.append(f"ADD  {_format_register(instr.dst)}, {_format_register(instr.src)}")
            case OpCode.MUL:
                lines.append(f"MUL  {_format_register(instr.dst)}, {_format_register(instr.src)}")
            case OpCode.SIGMOID:
                lines.append(f"SIGMOID {_format_register(instr.dst)}")
    return "\n".join(lines)


__all__ = ("assemble", "disassemble")