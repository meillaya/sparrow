// RUN: %forge-opt %s --action=lower | %FileCheck %s

// Idempotency golden test: feeding already-lowered IR (i.e. ops that live
// in `forge.tensor`, not `forge.graph`) through the GraphToTensor pass
// must be a structural no-op. The conversion patterns match only
// `forge.graph.*` ops, so the walker should leave every line of input
// untouched.
//
// FileCheck expectations:
//   * function name preserved
//   * the three tensor ops survive verbatim
//   * NO `forge.graph.*` ops appear (input has none, output has none)

// CHECK-LABEL: func.func @already_tensor
func.func @already_tensor(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK-NOT: forge.graph
    // CHECK: forge.tensor.gemm
    // CHECK-SAME: !forge.tensor<2x512x4096xf32>
    // CHECK-SAME: !forge.tensor<4096x11008xf32>
    %1 = forge.tensor.broadcast_add %0, %b
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.broadcast_add
    // CHECK-SAME: !forge.tensor<2x512x11008xf32>
    // CHECK-SAME: !forge.tensor<11008xf32>
    %2 = forge.tensor.gelu %1 {approximate = false}
        : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.gelu
    // CHECK-SAME: !forge.tensor<2x512x11008xf32>
    // CHECK: return
    return %2 : !forge.tensor<2x512x11008xf32>
}