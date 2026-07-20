// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// Round-trip golden for `forge.tensor.layer_norm`. Parses, prints, and
// FileCheck-verifies the structural skeleton (op present, epsilon attr
// present, operand / result types match).

// CHECK-LABEL: func.func @layer_norm_test
// CHECK-SAME: !forge.tensor<2x512xf32>
func.func @layer_norm_test(
    %x: !forge.tensor<2x512xf32>,
    %g: !forge.tensor<512xf32>,
    %b: !forge.tensor<512xf32>
) -> !forge.tensor<2x512xf32> {
    // CHECK: forge.tensor.layer_norm
    // CHECK-SAME: epsilon = 2.000000e-05 : f32
    %0 = forge.tensor.layer_norm %x, %g, %b {epsilon = 2.000000e-05 : f32}
        : (!forge.tensor<2x512xf32>, !forge.tensor<512xf32>, !forge.tensor<512xf32>) -> !forge.tensor<2x512xf32>
    return %0 : !forge.tensor<2x512xf32>
}