// RUN: %forge-opt %s --action=lower | %FileCheck %s

// Golden test for the GraphToTensor conversion pass. Mirrors the
// FusedMLP vertical slice from the plan: `matmul + bias + gelu` should
// lower 1:1 into `forge.tensor.{gemm, broadcast_add, gelu}` with no
// cross-op fusion.
//
// FileCheck expectations:
//   * function name preserved (@fused_mlp_lowered)
//   * NO `forge.graph.*` ops in the output
//   * the three tensor ops appear in source order
//   * a `return` is still present

// CHECK-LABEL: func.func @fused_mlp_lowered
func.func @fused_mlp_lowered(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.graph.matmul %x, %w : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK-NOT: forge.graph
    // CHECK: forge.tensor.gemm
    // CHECK-SAME: !forge.tensor<2x512x4096xf32>
    // CHECK-SAME: !forge.tensor<4096x11008xf32>
    %1 = forge.graph.add %0, %b : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.broadcast_add
    // CHECK-SAME: !forge.tensor<2x512x11008xf32>
    // CHECK-SAME: !forge.tensor<11008xf32>
    %2 = forge.graph.gelu %1 {approximate = "none"} : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.gelu
    // CHECK-SAME: !forge.tensor<2x512x11008xf32>
    // CHECK: return
    return %2 : !forge.tensor<2x512x11008xf32>
}