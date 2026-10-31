// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// The reshape verifier requires equal element counts and static shapes.

func.func @bad_reshape(%x: !forge.tensor<2x3xf32>) -> !forge.tensor<2x4xf32> {
    // CHECK-ERR: element count must match
    %0 = forge.tensor.reshape %x : (!forge.tensor<2x3xf32>) -> !forge.tensor<2x4xf32>
    return %0 : !forge.tensor<2x4xf32>
}
