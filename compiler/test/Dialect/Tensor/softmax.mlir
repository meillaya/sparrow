// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// Round-trip golden for `forge.tensor.softmax`. Parses, prints, and
// FileCheck-verifies the structural skeleton. The `dim` attr defaults
// to -1 (PyTorch convention), so the textual IR omits it.

// CHECK-LABEL: func.func @softmax_test
// CHECK-SAME: !forge.tensor<2x16xf32>
func.func @softmax_test(%x: !forge.tensor<2x16xf32>) -> !forge.tensor<2x16xf32> {
    // CHECK: forge.tensor.softmax
    %0 = forge.tensor.softmax %x {dim = #builtin.int<-1>}
        : (!forge.tensor<2x16xf32>) -> !forge.tensor<2x16xf32>
    return %0 : !forge.tensor<2x16xf32>
}