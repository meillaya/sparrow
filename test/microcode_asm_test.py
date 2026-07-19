"""Text assembler/disassembler tests for Astra microcode."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, runtime_checkable

import pytest
from xdsl.dialects.builtin import IntAttr, IntegerType, UnrealizedConversionCastOp
from xdsl.ir import SSAValue

from compiler.lib.Dialect.Astra import (
    ComputeMatmulOp,
    MemorySpace,
    MicrocodeDmaLoadOp,
    MicrocodeDmaStoreOp,
    MicrocodeMatmulOp,
    MicrocodeVectorGeluOp,
    PtrType,
)
from compiler.lib.Dialect.Astra.microcode import OpCode, decode, encode

type AstraOp = (
    ComputeMatmulOp
    | MicrocodeDmaLoadOp
    | MicrocodeDmaStoreOp
    | MicrocodeMatmulOp
    | MicrocodeVectorGeluOp
)


@runtime_checkable
class _AssemblerApi(Protocol):
    def assemble(self, text: str, /) -> list[int]: ...

    def disassemble(self, words: list[int], /) -> str: ...

    def assemble_from_dialect(self, microcode_ops: list[AstraOp], /) -> list[int]: ...

    def disassemble_to_dialect(self, words: list[int], /) -> list[AstraOp]: ...


_assembler_module = import_module("compiler.lib.Dialect.Astra.assembler")
assert isinstance(_assembler_module, _AssemblerApi)
assembler = _assembler_module

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
    assert assembler.assemble(text) == [expected]


@pytest.mark.parametrize(("text", "expected"), TEXT_CASES)
def test_disassemble_round_trip(text: str, expected: int) -> None:
    """Given one instruction word, disassembly reassembles to the same word."""
    words = assembler.assemble(text)

    assert assembler.disassemble(words) == assembler.disassemble([expected])
    assert assembler.assemble(assembler.disassemble(words)) == words


def test_assemble_ignores_blank_lines_and_comments() -> None:
    """Given formatting-only lines, assembly emits only executable operations."""
    text = "\n# DMA staging\nNOP\n\nHALT\n"

    assert assembler.assemble(text) == [
        encode(OpCode.NOP, 0, 0),
        encode(OpCode.HALT, 0, 0),
    ]


def test_assemble_rejects_malformed_lines() -> None:
    """Given an unknown or incomplete line, assembly reports its line number."""
    with pytest.raises(ValueError, match=r"line 1.*missing bytes, src"):
        _ = assembler.assemble("DMA_LOAD dst=SRAM[0x100]")


def _dialect_placeholders() -> tuple[SSAValue, SSAValue]:
    """Create detached SSA values suitable for standalone Astra ops."""
    dram = UnrealizedConversionCastOp.get([], [IntegerType(64)]).results[0]
    sram = UnrealizedConversionCastOp.get([], [PtrType(MemorySpace.SRAM0)]).results[0]
    return dram, sram


def test_dialect_round_trip_covers_all_five_astra_ops() -> None:
    """Given the five Astra dialect ops, bytes restore their op kinds and attrs."""
    dram, sram = _dialect_placeholders()
    sram_type = PtrType(MemorySpace.SRAM0)
    ops: list[AstraOp] = [
        MicrocodeDmaLoadOp(dram, sram_type, bytes=1024),
        MicrocodeDmaStoreOp(sram, IntegerType(64), bytes=1024),
        MicrocodeMatmulOp.create(
            operands=[sram, sram],
            result_types=[sram_type],
            properties={"m": IntAttr(16), "n": IntAttr(16), "k": IntAttr(16)},
        ),
        MicrocodeVectorGeluOp(sram, sram_type),
        ComputeMatmulOp(m=16, n=16, k=16),
    ]

    words = assembler.assemble_from_dialect(ops)
    decoded_ops = assembler.disassemble_to_dialect(words)

    assert [op.name for op in decoded_ops] == [
        "astra.microcode.dma_load",
        "astra.microcode.dma_store",
        "astra.microcode.matmul",
        "astra.microcode.vector_gelu",
        "astra.compute.matmul",
    ]
    assert [decode(word).opcode for word in words] == [
        OpCode.DMA_LOAD,
        OpCode.DMA_STORE,
        OpCode.MATMUL,
        OpCode.VECTOR_GELU,
        OpCode.MATMUL,
    ]
    match decoded_ops:
        case [
            MicrocodeDmaLoadOp() as dma_load,
            MicrocodeDmaStoreOp() as dma_store,
            MicrocodeMatmulOp() as matmul,
            MicrocodeVectorGeluOp() as gelu,
            ComputeMatmulOp() as compute,
        ]:
            assert dma_load.bytes.data == 1024
            assert dma_store.bytes.data == 1024
            assert matmul.m.data == 16
            assert gelu.sram_acc.type == sram.type
            assert compute.m.data == 16
        case _:
            pytest.fail("dialect round-trip returned unexpected operation types")
    assert assembler.assemble_from_dialect(decoded_ops) == words
