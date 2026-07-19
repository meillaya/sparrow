"""Text assembler and disassembler for the Astra microcode ISA."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from xdsl.dialects.builtin import IntAttr, IntegerType, UnrealizedConversionCastOp
from xdsl.ir import Attribute, SSAValue

from . import (
    ComputeMatmulOp,
    MemorySpace,
    MicrocodeDmaLoadOp,
    MicrocodeDmaStoreOp,
    MicrocodeMatmulOp,
    MicrocodeVectorGeluOp,
    PtrType,
)
from .microcode import MicrocodeInstr, OpCode, decode, encode

_FIELD_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<name>[a-zA-Z][a-zA-Z0-9_]*)=(?P<value>.+)$")
_ADDRESS_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<space>SRAM|DRAM)\[(?P<address>0[xX][0-9a-fA-F]+|[0-9]+)\]$",
    re.IGNORECASE,
)


def _line_error(line_number: int, message: str) -> ValueError:
    return ValueError(f"line {line_number}: {message}")


def _parse_fields(raw: str, line_number: int) -> Mapping[str, str]:
    if not raw.strip():
        return {}
    fields: dict[str, str] = {}
    for item in raw.split(","):
        match = _FIELD_RE.fullmatch(item.strip())
        if match is None:
            raise _line_error(line_number, f"malformed field {item.strip()!r}")
        name = match.group("name").lower()
        if name in fields:
            raise _line_error(line_number, f"duplicate field {name!r}")
        fields[name] = match.group("value").strip()
    return fields


def _require_fields(fields: Mapping[str, str], expected: frozenset[str], line_number: int) -> None:
    actual = frozenset(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise _line_error(line_number, "; ".join(details))


def _parse_number(value: str, field: str, line_number: int) -> int:
    try:
        number = int(value, 0)
    except ValueError as exc:
        raise _line_error(line_number, f"{field} must be an integer, got {value!r}") from exc
    if number < 0:
        raise _line_error(line_number, f"{field} must be non-negative")
    return number


def _parse_address(value: str, expected_space: str, field: str, line_number: int) -> int:
    match = _ADDRESS_RE.fullmatch(value)
    if match is None or match.group("space").upper() != expected_space:
        raise _line_error(
            line_number,
            f"{field} must be {expected_space}[address], got {value!r}",
        )
    return _parse_number(match.group("address"), field, line_number)


def _parse_instruction(line: str, line_number: int) -> int:
    opcode_name, separator, raw_fields = line.partition(" ")
    if not opcode_name or (not separator and raw_fields):
        raise _line_error(line_number, f"malformed instruction {line!r}")
    try:
        opcode = OpCode[opcode_name.upper()]
    except KeyError as exc:
        raise _line_error(line_number, f"unknown opcode {opcode_name!r}") from exc
    fields = _parse_fields(raw_fields, line_number)

    match opcode:
        case OpCode.NOP | OpCode.HALT:
            _require_fields(fields, frozenset(), line_number)
            return encode(opcode, 0, 0)
        case OpCode.DMA_LOAD:
            _require_fields(fields, frozenset({"dst", "src", "bytes"}), line_number)
            dst = _parse_address(fields["dst"], "SRAM", "dst", line_number)
            src = _parse_address(fields["src"], "DRAM", "src", line_number)
            size = _parse_number(fields["bytes"], "bytes", line_number)
            return encode(opcode, dst, src, imm=size)
        case OpCode.DMA_STORE:
            _require_fields(fields, frozenset({"dst", "src", "bytes"}), line_number)
            dst = _parse_address(fields["dst"], "DRAM", "dst", line_number)
            src = _parse_address(fields["src"], "SRAM", "src", line_number)
            size = _parse_number(fields["bytes"], "bytes", line_number)
            return encode(opcode, dst, src, imm=size)
        case OpCode.MATMUL:
            _require_fields(fields, frozenset({"a", "b", "k"}), line_number)
            a = _parse_address(fields["a"], "SRAM", "a", line_number)
            b = _parse_address(fields["b"], "SRAM", "b", line_number)
            k = _parse_number(fields["k"], "k", line_number)
            return encode(opcode, a, b, imm=k)
        case OpCode.VECTOR_GELU:
            _require_fields(fields, frozenset({"dst", "src", "elements"}), line_number)
            dst = _parse_address(fields["dst"], "SRAM", "dst", line_number)
            src = _parse_address(fields["src"], "SRAM", "src", line_number)
            elements = _parse_number(fields["elements"], "elements", line_number)
            return encode(opcode, dst, src, imm=elements)


def assemble(text: str) -> list[int]:
    """Assemble one textual Astra instruction per non-empty input line."""
    words: list[int] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if line:
            words.append(_parse_instruction(line, line_number))
    return words


def _format_address(space: str, address: int) -> str:
    return f"{space}[0x{address:x}]"


def disassemble(words: list[int]) -> str:
    """Render instruction words in the canonical Astra text format."""
    lines: list[str] = []
    for word in words:
        instr = decode(word)
        match instr.opcode:
            case OpCode.NOP | OpCode.HALT:
                lines.append(instr.opcode.name)
            case OpCode.DMA_LOAD:
                lines.append(
                    f"DMA_LOAD  dst={_format_address('SRAM', instr.reg_a)}, src={_format_address('DRAM', instr.reg_b)}, bytes={instr.imm}"
                )
            case OpCode.DMA_STORE:
                lines.append(
                    f"DMA_STORE dst={_format_address('DRAM', instr.reg_a)}, src={_format_address('SRAM', instr.reg_b)}, bytes={instr.imm}"
                )
            case OpCode.MATMUL:
                lines.append(
                    f"MATMUL a={_format_address('SRAM', instr.reg_a)}, b={_format_address('SRAM', instr.reg_b)}, k={instr.imm}"
                )
            case OpCode.VECTOR_GELU:
                lines.append(
                    f"VECTOR_GELU dst={_format_address('SRAM', instr.reg_a)}, src={_format_address('SRAM', instr.reg_b)}, elements={instr.imm}"
                )
    return "\n".join(lines)


type AstraOp = (
    ComputeMatmulOp
    | MicrocodeDmaLoadOp
    | MicrocodeDmaStoreOp
    | MicrocodeMatmulOp
    | MicrocodeVectorGeluOp
)

_DIALECT_MARKER = 0x8000
_DIALECT_VALUE_MASK = _DIALECT_MARKER - 1


def _encode_dialect_op(op: AstraOp) -> int:
    match op:
        case ComputeMatmulOp():
            return encode(
                OpCode.MATMUL,
                op.m.data | _DIALECT_MARKER,
                op.n.data,
                imm=op.k.data,
            )
        case MicrocodeDmaLoadOp():
            return encode(OpCode.DMA_LOAD, 0, 0, imm=op.bytes.data)
        case MicrocodeDmaStoreOp():
            return encode(OpCode.DMA_STORE, 0, 0, imm=op.bytes.data)
        case MicrocodeMatmulOp():
            bias_marker = _DIALECT_MARKER if op.bias is not None else 0
            return encode(
                OpCode.MATMUL,
                op.m.data,
                op.n.data | bias_marker,
                imm=op.k.data,
            )
        case MicrocodeVectorGeluOp():
            return encode(OpCode.VECTOR_GELU, 0, 0)


def assemble_from_dialect(microcode_ops: list[AstraOp]) -> list[int]:
    """Encode the supported Astra dialect operations in list order."""
    return [_encode_dialect_op(op) for op in microcode_ops]


def _placeholder(attribute: Attribute) -> SSAValue:
    return UnrealizedConversionCastOp.get([], [attribute]).results[0]


def _decode_dialect_op(instr: MicrocodeInstr) -> AstraOp:
    match instr.opcode:
        case OpCode.DMA_LOAD:
            return MicrocodeDmaLoadOp(
                _placeholder(IntegerType(64)),
                PtrType(MemorySpace.SRAM0),
                bytes=instr.imm,
            )
        case OpCode.DMA_STORE:
            return MicrocodeDmaStoreOp(
                _placeholder(PtrType(MemorySpace.SRAM0)),
                IntegerType(64),
                bytes=instr.imm,
            )
        case OpCode.MATMUL:
            if instr.reg_a & _DIALECT_MARKER:
                return ComputeMatmulOp(
                    m=instr.reg_a & _DIALECT_VALUE_MASK,
                    n=instr.reg_b,
                    k=instr.imm,
                )
            sram_type = PtrType(MemorySpace.SRAM0)
            operands = [
                _placeholder(sram_type),
                _placeholder(sram_type),
            ]
            if instr.reg_b & _DIALECT_MARKER:
                operands.append(_placeholder(sram_type))
            return MicrocodeMatmulOp.create(
                operands=operands,
                result_types=[sram_type],
                properties={
                    "m": IntAttr(instr.reg_a),
                    "n": IntAttr(instr.reg_b & _DIALECT_VALUE_MASK),
                    "k": IntAttr(instr.imm),
                },
            )
        case OpCode.VECTOR_GELU:
            sram = PtrType(MemorySpace.SRAM0)
            return MicrocodeVectorGeluOp(
                _placeholder(sram),
                sram,
            )
        case OpCode.NOP | OpCode.HALT:
            raise ValueError(f"{instr.opcode.name} has no Astra dialect operation")


def disassemble_to_dialect(words: list[int]) -> list[AstraOp]:
    """Decode words into detached, verified-shape Astra dialect operations."""
    return [_decode_dialect_op(decode(word)) for word in words]
