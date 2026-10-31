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
the word stream modulo whitespace and comments. The dialect bridge
(:func:`assemble_from_microgpu_dialect` /
:func:`disassemble_to_microgpu_dialect`) lets the rest of the
compiler drive the assembler from the typed
:mod:`compiler.lib.Dialect.MicroGPU` ops via the lightweight
:class:`MicrogpuOp` record (one field per wire-format slot).

Reference: spec §11. Mirrors
:mod:`compiler.lib.Dialect.Astra.assembler` in shape.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from backends.microgpu.isa import OpCode, decode, encode

# Register literals ("r0".."r15") parsed from the textual assembly.
# Out-of-range register numbers are rejected by ``_parse_register``.
_REGISTER_RE: Final[re.Pattern[str]] = re.compile(r"^r(?P<num>[0-9]+)$")

# Memory operand "[0xADDR]" or "[DEC]" parsed from the textual assembly.
_MEMORY_RE: Final[re.Pattern[str]] = re.compile(r"^\[(?P<addr>0[xX][0-9a-fA-F]+|[0-9]+)\]$")


MicrogpuOpName = Literal[
    "load_global",
    "store_global",
    "add",
    "mul",
    "sigmoid",
    "matmul",
    "fadd",
    "fmul",
    "frelu",
    "return",
]


@dataclass(frozen=True, slots=True)
class MicrogpuOp:
    """One MicroGPU dialect operation in the assembler's detached-record form.

    Each field corresponds to one wire-format slot (see
    :mod:`backends.microgpu.isa`):

    * ``name`` is the dialect op kind (``microgpu.<name>`` in xDSL).
    * ``dst`` is the 4-bit destination register slot (0..15).
    * ``src`` is the 4-bit source register slot (0..15); unused for
      load_global / return.
    * ``imm`` is the 4-bit immediate / address (0..15); unused for
      add / mul / sigmoid / return.

    Records are detached: they carry no SSA values or module context,
    so the assembler can build them from a textual program or from a
    xDSL op without coupling to either source. The bridge functions
    :func:`assemble_from_microgpu_dialect` and
    :func:`disassemble_to_microgpu_dialect` are the canonical
    serialisation for these records.
    """

    name: MicrogpuOpName
    dst: int = 0
    src: int = 0
    imm: int = 0


def _line_error(line_number: int, message: str) -> ValueError:
    """Build a line-numbered :class:`ValueError` for assembly errors."""
    return ValueError(f"line {line_number}: {message}")


def _parse_register(token: str, line_number: int) -> int:
    """Parse a textual register reference (``r0``..``r15``) into a 4-bit slot."""
    match = _REGISTER_RE.fullmatch(token)
    if match is None:
        raise _line_error(line_number, f"expected register like r0..r15, got {token!r}")
    number = int(match.group("num"))
    if number < 0 or number > 0x1F:
        raise _line_error(line_number, f"register {token!r} does not fit the 32-slot file")
    return number


def _parse_address(token: str, line_number: int) -> int:
    """Parse a bracketed memory operand (``[0x1000]``) into an integer address.

    The 32-bit word carries a 16-bit address, so a program may address
    up to 65,536 words (256 KiB).
    """
    match = _MEMORY_RE.fullmatch(token)
    if match is None:
        raise _line_error(line_number, f"expected bracketed address like [0x1000], got {token!r}")
    try:
        address = int(match.group("addr"), 0)
    except ValueError as exc:
        raise _line_error(line_number, f"address must be an integer, got {token!r}") from exc
    if address < 0 or address > 0xFFFF:
        raise _line_error(line_number, f"address {token!r} does not fit in 16-bit immediate")
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
        case OpCode.SIGMOID | OpCode.FRELU:
            if len(operands) != 1:
                raise _line_error(
                    line_number,
                    f"{opcode.name} expects 1 operand (dst), got {operands!r}",
                )
            dst = _parse_register(operands[0], line_number)
            return encode(opcode, dst, 0, imm=0)
        case OpCode.FADD | OpCode.FMUL:
            if len(operands) != 2:
                raise _line_error(
                    line_number,
                    f"{opcode.name} expects 2 operands (dst, src), got {operands!r}",
                )
            dst = _parse_register(operands[0], line_number)
            src = _parse_register(operands[1], line_number)
            return encode(opcode, dst, src, imm=0)
        case OpCode.MATMUL:
            if len(operands) != 1:
                raise _line_error(
                    line_number,
                    f"MATMUL expects 1 operand ([base]), got {operands!r}",
                )
            base = _parse_address(operands[0], line_number)
            return encode(opcode, 0, 0, imm=base)


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
            case OpCode.FADD:
                lines.append(f"FADD {_format_register(instr.dst)}, {_format_register(instr.src)}")
            case OpCode.FMUL:
                lines.append(f"FMUL {_format_register(instr.dst)}, {_format_register(instr.src)}")
            case OpCode.FRELU:
                lines.append(f"FRELU {_format_register(instr.dst)}")
            case OpCode.MATMUL:
                lines.append(f"MATMUL {_format_address(instr.imm)}")
    return "\n".join(lines)


def _encode_microgpu_op(op: MicrogpuOp) -> int:
    """Translate one :class:`MicrogpuOp` record into a 16-bit instruction word.

    Validates that each populated field fits the 4-bit wire range and
    that the op kind is one of the seven dialect-supported names.
    """
    match op.name:
        case "load_global":
            return encode(OpCode.LOAD_GLOBAL, op.dst, 0, imm=op.imm)
        case "store_global":
            return encode(OpCode.STORE_GLOBAL, 0, op.src, imm=op.imm)
        case "add":
            return encode(OpCode.ADD, op.dst, op.src, imm=0)
        case "mul":
            return encode(OpCode.MUL, op.dst, op.src, imm=0)
        case "sigmoid":
            return encode(OpCode.SIGMOID, op.dst, 0, imm=0)
        case "fadd":
            return encode(OpCode.FADD, op.dst, op.src, imm=0)
        case "fmul":
            return encode(OpCode.FMUL, op.dst, op.src, imm=0)
        case "frelu":
            return encode(OpCode.FRELU, op.dst, 0, imm=0)
        case "matmul":
            return encode(OpCode.MATMUL, 0, 0, imm=op.imm)
        case "return":
            return encode(OpCode.RETURN, 0, 0, imm=0)
        case _:
            raise ValueError(f"unknown microgpu dialect op: {op.name!r}")


def assemble_from_microgpu_dialect(microgpu_ops: Sequence[MicrogpuOp]) -> list[int]:
    """Encode the supported MicroGPU dialect operations in list order."""
    return [_encode_microgpu_op(op) for op in microgpu_ops]


def _decode_microgpu_op(word: int) -> MicrogpuOp:
    """Translate one 16-bit instruction word into a :class:`MicrogpuOp` record."""
    instr = decode(word)
    match instr.opcode:
        case OpCode.LOAD_GLOBAL:
            return MicrogpuOp(name="load_global", dst=instr.dst, imm=instr.imm)
        case OpCode.STORE_GLOBAL:
            return MicrogpuOp(name="store_global", src=instr.src, imm=instr.imm)
        case OpCode.ADD:
            return MicrogpuOp(name="add", dst=instr.dst, src=instr.src)
        case OpCode.MUL:
            return MicrogpuOp(name="mul", dst=instr.dst, src=instr.src)
        case OpCode.SIGMOID:
            return MicrogpuOp(name="sigmoid", dst=instr.dst)
        case OpCode.FADD:
            return MicrogpuOp(name="fadd", dst=instr.dst, src=instr.src)
        case OpCode.FMUL:
            return MicrogpuOp(name="fmul", dst=instr.dst, src=instr.src)
        case OpCode.FRELU:
            return MicrogpuOp(name="frelu", dst=instr.dst)
        case OpCode.MATMUL:
            return MicrogpuOp(name="matmul", imm=instr.imm)
        case OpCode.RETURN:
            return MicrogpuOp(name="return")
        case OpCode.NOP:
            raise ValueError("NOP has no MicroGPU dialect op")
        case _:
            raise ValueError(f"unhandled wire opcode {instr.opcode!r}")


def disassemble_to_microgpu_dialect(words: Sequence[int]) -> list[MicrogpuOp]:
    """Decode instruction words into detached :class:`MicrogpuOp` records."""
    return [_decode_microgpu_op(word) for word in words]


__all__ = (
    "MicrogpuOp",
    "MicrogpuOpName",
    "assemble",
    "assemble_from_microgpu_dialect",
    "disassemble",
    "disassemble_to_microgpu_dialect",
)
