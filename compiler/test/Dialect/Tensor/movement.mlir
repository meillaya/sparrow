// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// Round-trip golden for the M3 element-wise and movement ops:
// relu, mul, transpose, select.

// CHECK-LABEL: func.func @elementwise
func.func @elementwise(
    %x: !forge.tensor<2x3xf32>,
    %s: !forge.tensor<1xf32>
) -> !forge.tensor<2x3xf32> {
    // CHECK: forge.tensor.relu
    %0 = forge.tensor.relu %x : (!forge.tensor<2x3xf32>) -> !forge.tensor<2x3xf32>
    // CHECK: forge.tensor.mul
    %1 = forge.tensor.mul %0, %s : (!forge.tensor<2x3xf32>, !forge.tensor<1xf32>) -> !forge.tensor<2x3xf32>
    return %1 : !forge.tensor<2x3xf32>
}

// CHECK-LABEL: func.func @movement
func.func @movement(%x: !forge.tensor<2x3x4xf32>) -> !forge.tensor<4x2x3xf32> {
    // CHECK: forge.tensor.transpose
    // CHECK-SAME: permutation = [#builtin.int<2>, #builtin.int<0>, #builtin.int<1>]
    %0 = forge.tensor.transpose %x {permutation = [#builtin.int<2>, #builtin.int<0>, #builtin.int<1>]}
        : (!forge.tensor<2x3x4xf32>) -> !forge.tensor<4x2x3xf32>
    return %0 : !forge.tensor<4x2x3xf32>
}

// CHECK-LABEL: func.func @select
func.func @select(%x: !forge.tensor<3x2x4xf32>) -> !forge.tensor<2x4xf32> {
    // CHECK: forge.tensor.select
    // CHECK-SAME: dim = #builtin.int<0>
    // CHECK-SAME: index = #builtin.int<2>
    %0 = forge.tensor.select %x {dim = #builtin.int<0>, index = #builtin.int<2>}
        : (!forge.tensor<3x2x4xf32>) -> !forge.tensor<2x4xf32>
    return %0 : !forge.tensor<2x4xf32>
}
