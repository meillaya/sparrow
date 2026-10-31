// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s
//
// Negative test: AddOp.verify_ must reject a result dtype that differs
// from the (agreeing) operand dtypes, even when the broadcast shape and
// the result shape are valid.

// CHECK-ERR: error: add result dtype f16 does not match operand dtype f32
func.func @bad_add_result_dtype(
    %a: !forge.tensor<2x512x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf16> {
    %0 = forge.graph.add %a, %b : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf16>
    return %0 : !forge.tensor<2x512x11008xf16>
}
