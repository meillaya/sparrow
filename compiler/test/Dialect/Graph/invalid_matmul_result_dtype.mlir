// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s
//
// Negative test: MatmulOp.verify_ must reject a result dtype that differs
// from the (agreeing) operand dtypes, even when the result shape is valid.

// CHECK-ERR: error: matmul result dtype f16 does not match operand dtype f32
func.func @bad_matmul_result_dtype(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>
) -> !forge.tensor<2x512x11008xf16> {
    %0 = forge.graph.matmul %x, %w : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf16>
    return %0 : !forge.tensor<2x512x11008xf16>
}
