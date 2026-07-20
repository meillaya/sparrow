// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// Round-trip golden for `forge.tensor.scaled_dot_product_attention`.
// Parses, prints, and FileCheck-verifies the structural skeleton. The
// fused op takes Q, K, V of matching shape and returns the same shape
// (this is the canonical Phase 7 attention primitive).

// CHECK-LABEL: func.func @sdpa_test
// CHECK-SAME: !forge.tensor<2x16x64xf32>
func.func @sdpa_test(
    %q: !forge.tensor<2x16x64xf32>,
    %k: !forge.tensor<2x16x64xf32>,
    %v: !forge.tensor<2x16x64xf32>
) -> !forge.tensor<2x16x64xf32> {
    // CHECK: forge.tensor.scaled_dot_product_attention
    // CHECK-SAME: num_heads = #builtin.int<4>
    %0 = forge.tensor.scaled_dot_product_attention %q, %k, %v {num_heads = #builtin.int<4>}
        : (!forge.tensor<2x16x64xf32>, !forge.tensor<2x16x64xf32>, !forge.tensor<2x16x64xf32>) -> !forge.tensor<2x16x64xf32>
    return %0 : !forge.tensor<2x16x64xf32>
}