// RUN: not %forge-opt %s --split-input-file --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test 1: forge.tensor.gemm must reject shapes whose inner K dim
// does not match (no implicit broadcasting on gemm).

// CHECK-ERR: gemm inner dims mismatch
func.func @bad_gemm(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<2048x11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<2048x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %0 : !forge.tensor<2x512x11008xf32>
}

// -----

// Negative test 2: forge.tensor.fused_epilogue must reject an activation
// value that isn't one of the four enum entries. The xDSL ParseError
// pretty-prints the message on its own line (after the source location
// caret block), so we match the unique substring rather than an
// "error: …" prefix.

// CHECK-ERR: unknown activation 'bogus'
func.func @bad_activation(
    %x: !forge.tensor<2x512xf32>
) -> !forge.tensor<2x512xf32> {
    %0 = forge.tensor.fused_epilogue %x {activation = #forge.activation<bogus>} : (!forge.tensor<2x512xf32>) -> !forge.tensor<2x512xf32>
    return %0 : !forge.tensor<2x512xf32>
}
