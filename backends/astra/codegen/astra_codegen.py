"""emit_astra — lower an Astra microcode region to instruction words.

The Phase 4 codegen is deliberately small: it discovers ``astra.microcode.*``
operations in textual xDSL IR, preserves byte-count and ``k`` attributes that
fit the current instruction word, and delegates packing to the assembler.

Reference: spec §9 and Wave 11 Task 48.
"""

from __future__ import annotations

import re
from importlib import import_module
from typing import Final, Protocol, runtime_checkable

from compiler.lib.Dialect.Astra.microcode import OpCode


@runtime_checkable
class _AssemblerApi(Protocol):
    def assemble(self, text: str, /) -> list[int]: ...


_assembler_module = import_module("compiler.lib.Dialect.Astra.assembler")
if isinstance(_assembler_module, _AssemblerApi):
    _ASSEMBLER: Final = _assembler_module
else:
    raise TypeError("Astra assembler module does not provide assemble(text)")


_OP_RE: Final[re.Pattern[str]] = re.compile(r"\bastra\.microcode\.(?P<opcode>[a-z][a-z0-9_]*)\b")


def _integer_attribute(line: str, name: str, default: int = 0) -> int:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(?:#builtin\.int<)?(?P<value>[0-9]+)>?",
        line,
    )
    return default if match is None else int(match.group("value"))


def _assembly_line(opcode: OpCode, region_line: str) -> str:
    match opcode:
        case OpCode.DMA_LOAD:
            size = _integer_attribute(region_line, "bytes")
            return f"DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes={size}"
        case OpCode.DMA_STORE:
            size = _integer_attribute(region_line, "bytes")
            return f"DMA_STORE dst=DRAM[0x0], src=SRAM[0x0], bytes={size}"
        case OpCode.MATMUL:
            k = _integer_attribute(region_line, "k")
            return f"MATMUL a=SRAM[0x0], b=SRAM[0x0], k={k}"
        case OpCode.VECTOR_GELU:
            return "VECTOR_GELU dst=SRAM[0x0], src=SRAM[0x0], elements=0"
        case OpCode.NOP | OpCode.HALT:
            return opcode.name


def emit_astra(region: str) -> list[int]:
    """Emit one packed Astra instruction for each microcode op in ``region``."""
    assembly_lines: list[str] = []
    for line_number, line in enumerate(region.splitlines(), start=1):
        for operation_match in _OP_RE.finditer(line):
            opcode_name = operation_match.group("opcode").upper()
            try:
                opcode = OpCode[opcode_name]
            except KeyError as exc:
                raise ValueError(
                    f"line {line_number}: unsupported astra.microcode.{opcode_name.lower()}"
                ) from exc
            assembly_lines.append(_assembly_line(opcode, line))
    return _ASSEMBLER.assemble("\n".join(assembly_lines))
