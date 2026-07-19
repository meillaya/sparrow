// RUN: %forge-opt %s --action=lower-astra | %FileCheck %s

// Golden test for the TensorToAstra conversion pass (Wave 9 / Task 42).
// Mirrors the Phase-2-fused FusedMLP IR:
//
//     %y = forge.tensor.gemm %x, %w              (matmul only, no bias)
//     %z = forge.tensor.fused_epilogue %y, %b
//           {activation = #forge.activation<gelu>}    (bias + gelu)
//
// …which the conversion pass lowers to the canonical astra sequence
// (DMA loads → systolic matmul → vector_gelu → DMA store).
//
// FileCheck expectations:
//   * function name preserved (@fused_mlp_lowered)
//   * NO `forge.tensor.*` ops remain (everything lowered)
//   * the expected astra ops appear in source order
//   * DRAM operands are `i64`, SRAM operands carry the typed
//     `!astra.ptr<#astra.memory_space<sram0>>`

// CHECK-LABEL: func.func @fused_mlp_lowered
func.func @fused_mlp_lowered(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %y = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %z = forge.tensor.fused_epilogue %y, %b {activation = #forge.activation<gelu>}
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK-NOT: forge.tensor
    // CHECK: arith.constant
    // CHECK: astra.microcode.dma_load
    // CHECK-SAME: i64
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK: astra.microcode.matmul
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK-SAME: -> !astra.ptr<#astra.memory_space<sram0>>
    // CHECK: astra.microcode.vector_gelu
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK: astra.microcode.dma_store
    // CHECK-SAME: !astra.ptr<#astra.memory_space<sram0>>
    // CHECK-SAME: i64
    return %z : !forge.tensor<2x512x11008xf32>
}
