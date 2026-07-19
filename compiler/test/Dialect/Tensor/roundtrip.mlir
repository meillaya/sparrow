// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s
// RUN: %forge-opt %s --action=roundtrip | %forge-opt --no-auto-dialects --load-dialect compiler.lib.Dialect.Graph --load-dialect compiler.lib.Dialect.Tensor --action=roundtrip

// Round-trip golden test for forge.tensor: parse a small `fused_mlp_tensor`
// function (gemm + bias + activation fused), verify it, re-print it, and
// FileCheck the structural skeleton. The second RUN line is a stability
// guard — it re-parses the round-trip output through a fresh driver that
// loads both dialects by FQN, to make sure the output is itself parseable.

// CHECK-LABEL: func.func @fused_mlp_tensor
// CHECK-SAME: !forge.tensor<2x512x4096xf32>
// CHECK: forge.tensor.gemm
// CHECK-SAME: !forge.tensor<2x512x4096xf32>
// CHECK: forge.tensor.fused_epilogue
// CHECK-SAME: activation = #forge.activation<gelu>
// CHECK: return {{.*}} : !forge.tensor<2x512x11008xf32>
func.func @fused_mlp_tensor(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %1 = forge.tensor.fused_epilogue %0, %b {activation = #forge.activation<gelu>}
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %1 : !forge.tensor<2x512x11008xf32>
}
