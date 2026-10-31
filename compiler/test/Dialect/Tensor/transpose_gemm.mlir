// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// A linear layer lowers to x @ weight^T, so the verifier must apply the
// transpose flags before checking the inner and output dimensions.

// CHECK-LABEL: func.func @linear_shape
// CHECK: forge.tensor.gemm
// CHECK-SAME: transpose_b = true
func.func @linear_shape(%x: !forge.tensor<2x8xf32>, %w: !forge.tensor<16x8xf32>) -> !forge.tensor<2x16xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_b = true} : (!forge.tensor<2x8xf32>, !forge.tensor<16x8xf32>) -> !forge.tensor<2x16xf32>
    return %0 : !forge.tensor<2x16xf32>
}
