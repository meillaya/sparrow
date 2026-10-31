"""Astra region-to-microcode codegen tests."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, runtime_checkable

from compiler.lib.Dialect.Astra.microcode import OpCode, decode


@runtime_checkable
class _AstraCodegenApi(Protocol):
    def emit_astra(self, region: str, /) -> list[int]: ...


_codegen_module = import_module("backends.astra.codegen.astra_codegen")
assert isinstance(_codegen_module, _AstraCodegenApi)
codegen = _codegen_module

fused_mlp_microcode_region = """
func.func @region(%dram_a: i64, %dram_b: i64) -> i64 {
  %s0 = astra.microcode.dma_load %dram_a {bytes = #builtin.int<8192>, sram_offset = #builtin.int<0>}
      : (i64) -> !astra.ptr<#astra.memory_space<sram0>>
  %s1 = astra.microcode.dma_load %dram_b {bytes = #builtin.int<4096>, sram_offset = #builtin.int<8192>}
      : (i64) -> !astra.ptr<#astra.memory_space<sram0>>
  %s2 = astra.microcode.matmul %s0, %s1 {m = #builtin.int<16>, n = #builtin.int<16>, k = #builtin.int<16>}
      : (!astra.ptr<#astra.memory_space<sram0>>, !astra.ptr<#astra.memory_space<sram0>>)
        -> !astra.ptr<#astra.memory_space<sram0>>
  %s3 = astra.microcode.vector_gelu %s2 {elements = #builtin.int<256>, src_offset = #builtin.int<0>, dst_offset = #builtin.int<1024>}
      : (!astra.ptr<#astra.memory_space<sram0>>) -> !astra.ptr<#astra.memory_space<sram0>>
  %out = astra.microcode.dma_store %s3 {bytes = #builtin.int<4096>, dram_offset = #builtin.int<0>, sram_offset = #builtin.int<1024>}
      : (!astra.ptr<#astra.memory_space<sram0>>) -> i64
  return %out : i64
}
"""


def test_emit_astra_word_count_matches_microcode_op_count() -> None:
    """Given a fused-MLP region, codegen emits one word per microcode op."""
    words = codegen.emit_astra(fused_mlp_microcode_region)

    assert len(words) == fused_mlp_microcode_region.count("astra.microcode.")
    assert [decode(word).opcode for word in words] == [
        OpCode.DMA_LOAD,
        OpCode.DMA_LOAD,
        OpCode.MATMUL,
        OpCode.VECTOR_GELU,
        OpCode.DMA_STORE,
    ]
    assert decode(words[0]).reg_a == 0
    assert decode(words[0]).reg_b == 0
    assert decode(words[1]).reg_a == 8192
    assert decode(words[4]).reg_a == 0
    assert decode(words[4]).reg_b == 1024
