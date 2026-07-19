// RUN: not %forge-opt %s --split-input-file --pass-pipeline='shape-inference' 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test 1: a `forge.tensor.broadcast_add` whose operand shapes
// are not NumPy-broadcastable must be rejected by the dialect verifier
// before the shape-inference pass gets a chance to run. The substring
// (`operand shapes are not broadcastable`) is emitted by the tensor
// dialect's verifier in `compiler/lib/Dialect/Tensor/__init__.py`.

// CHECK-ERR: operand shapes are not broadcastable
func.func @bad_broadcast_shapes(
    %x: !forge.tensor<2x3xf32>,
    %y: !forge.tensor<4x5xf32>
) -> !forge.tensor<2x3xf32> {
    %0 = forge.tensor.broadcast_add %x, %y
        : (!forge.tensor<2x3xf32>, !forge.tensor<4x5xf32>) -> !forge.tensor<2x3xf32>
    return %0 : !forge.tensor<2x3xf32>
}

// -----

// Negative test 2: a `forge.tensor.gemm` with mismatched inner K dims
// is rejected by the verifier before the shape-inference pass runs.

// CHECK-ERR: gemm inner dims mismatch
func.func @bad_gemm_inner_k(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<2048x11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<2048x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %0 : !forge.tensor<2x512x11008xf32>
}
