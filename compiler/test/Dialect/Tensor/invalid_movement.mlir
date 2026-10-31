// RUN: not %forge-opt %s --split-input-file --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative tests for the M3 movement ops.

// CHECK-ERR: permutation must be a permutation of range(2)
func.func @bad_transpose(%x: !forge.tensor<2x3xf32>) -> !forge.tensor<3x2xf32> {
    %0 = forge.tensor.transpose %x {permutation = [#builtin.int<0>, #builtin.int<0>]}
        : (!forge.tensor<2x3xf32>) -> !forge.tensor<3x2xf32>
    return %0 : !forge.tensor<3x2xf32>
}

// -----

// CHECK-ERR: result shape mismatch
func.func @bad_transpose_shape(%x: !forge.tensor<2x3xf32>) -> !forge.tensor<2x3xf32> {
    %0 = forge.tensor.transpose %x {permutation = [#builtin.int<1>, #builtin.int<0>]}
        : (!forge.tensor<2x3xf32>) -> !forge.tensor<2x3xf32>
    return %0 : !forge.tensor<2x3xf32>
}

// -----

// CHECK-ERR: index 5 is out of range
func.func @bad_select_index(%x: !forge.tensor<3x2xf32>) -> !forge.tensor<2xf32> {
    %0 = forge.tensor.select %x {dim = #builtin.int<0>, index = #builtin.int<5>}
        : (!forge.tensor<3x2xf32>) -> !forge.tensor<2xf32>
    return %0 : !forge.tensor<2xf32>
}

// -----

// CHECK-ERR: not broadcastable
func.func @bad_mul(%x: !forge.tensor<2x3xf32>, %y: !forge.tensor<4xf32>) -> !forge.tensor<2x4xf32> {
    %0 = forge.tensor.mul %x, %y : (!forge.tensor<2x3xf32>, !forge.tensor<4xf32>) -> !forge.tensor<2x4xf32>
    return %0 : !forge.tensor<2x4xf32>
}

// -----

// CHECK-ERR: result shape must equal input shape
func.func @bad_relu(%x: !forge.tensor<2x3xf32>) -> !forge.tensor<2x4xf32> {
    %0 = forge.tensor.relu %x : (!forge.tensor<2x3xf32>) -> !forge.tensor<2x4xf32>
    return %0 : !forge.tensor<2x4xf32>
}
