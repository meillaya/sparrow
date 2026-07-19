// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// Round-trip golden test for the `astra` dialect (Wave 9 / Task 40):
// parse each of the 5 ops + the `MemorySpace` enum attr + the `PtrType`
// parametric type, verify, re-print, and FileCheck the structural
// skeleton. M / N / K / bytes ride as `IntAttr` properties (canonical
// `#builtin.int<...>` syntax in the attr-dict).
//
// FileCheck expectations:
//   * function name preserved (@astra_primitives)
//   * all 5 ops appear in the printed IR
//   * pointer types carry the typed `!astra.ptr<#astra.memory_space<...>>` syntax
//   * DRAM/SRAM distinction is statically visible in the type

// CHECK-LABEL: func.func @astra_primitives
func.func @astra_primitives(
    %dram_a: i64,
    %dram_b: i64,
    %dram_y: i64,
    %sram_a: !astra.ptr<#astra.memory_space<sram0>>,
    %sram_b: !astra.ptr<#astra.memory_space<sram0>>,
    %sram_c: !astra.ptr<#astra.memory_space<sram0>>
) -> i64 {
    // High-level compute command — marker only, M/N/K as attrs.
    // CHECK: astra.compute.matmul
    // CHECK-SAME: m = #builtin.int<16>
    // CHECK-SAME: n = #builtin.int<16>
    // CHECK-SAME: k = #builtin.int<16>
    astra.compute.matmul {m = #builtin.int<16>, n = #builtin.int<16>, k = #builtin.int<16>}

    // CHECK: astra.microcode.dma_load
    // CHECK-SAME: i64
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    %s0 = astra.microcode.dma_load %dram_a {bytes = #builtin.int<4096>} : (i64) -> !astra.ptr<#astra.memory_space<sram0>>

    // CHECK: astra.microcode.matmul
    // CHECK-SAME: m = #builtin.int<16>
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK-SAME: -> !astra.ptr<#astra.memory_space<sram0>>
    %s2 = astra.microcode.matmul %s0, %s0 {m = #builtin.int<16>, n = #builtin.int<16>, k = #builtin.int<16>} : (!astra.ptr<#astra.memory_space<sram0>>, !astra.ptr<#astra.memory_space<sram0>>) -> !astra.ptr<#astra.memory_space<sram0>>

    // CHECK: astra.microcode.vector_gelu
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    %s3 = astra.microcode.vector_gelu %s2 : (!astra.ptr<#astra.memory_space<sram0>>) -> !astra.ptr<#astra.memory_space<sram0>>

    // CHECK: astra.microcode.dma_store
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK-SAME: i64
    %y = astra.microcode.dma_store %s3 {bytes = #builtin.int<1024>} : (!astra.ptr<#astra.memory_space<sram0>>) -> i64

    // CHECK: return
    return %y : i64
}
