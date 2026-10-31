// RUN: %forge-opt %s --pass-pipeline='fuse' --action=optimize | %FileCheck %s
//
// Negative-shape (valid-IR) test for the fusion guard: when the
// broadcast_add result has a second consumer besides the bare activation,
// the pattern must not fire — erasing the add would leave a live SSA value
// behind and crash with a ValueError.

// CHECK-LABEL: func.func @multi_use_add_is_not_fused
// CHECK: forge.tensor.gemm
// CHECK: forge.tensor.broadcast_add
// CHECK-COUNT-2: forge.tensor.fused_epilogue
// CHECK: func.return
func.func @multi_use_add_is_not_fused(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> (!forge.tensor<2x512x11008xf32>, !forge.tensor<2x512x11008xf32>) {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %1 = forge.tensor.broadcast_add %0, %b
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %2 = forge.tensor.fused_epilogue %1 {activation = #forge.activation<gelu>}
        : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %3 = forge.tensor.fused_epilogue %1 {activation = #forge.activation<relu>}
        : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %2, %3 : !forge.tensor<2x512x11008xf32>, !forge.tensor<2x512x11008xf32>
}
