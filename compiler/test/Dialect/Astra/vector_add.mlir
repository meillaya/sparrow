// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// M5: the vector add carries an element count, a broadcast flag, and
// explicit SRAM offsets so the codegen can emit it without guessing.

// CHECK-LABEL: func.func @vector_add
func.func @vector_add(
    %sram_acc: !astra.ptr<#astra.memory_space<sram0>>,
    %sram_bias: !astra.ptr<#astra.memory_space<sram0>>
) -> !astra.ptr<#astra.memory_space<sram0>> {
    // CHECK: astra.microcode.vector_add
    // CHECK-SAME: elements = #builtin.int<256>
    // CHECK-SAME: broadcast = true
    // CHECK-SAME: src_offset = #builtin.int<3072>
    %0 = astra.microcode.vector_add %sram_acc, %sram_bias {
        elements = #builtin.int<256>,
        broadcast = true,
        dst_offset = #builtin.int<0>,
        src_offset = #builtin.int<3072>
    } : (!astra.ptr<#astra.memory_space<sram0>>, !astra.ptr<#astra.memory_space<sram0>>)
        -> !astra.ptr<#astra.memory_space<sram0>>
    return %0 : !astra.ptr<#astra.memory_space<sram0>>
}
