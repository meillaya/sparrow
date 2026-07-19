// RUN: not %forge-opt %s --pass-pipeline='fuse' --action=optimize 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test: an input where the dialect verifier fails BEFORE the
// fusion pass gets a chance to run. Specifically, the bias shape
// doesn't broadcast over the gemm output — the dialect verifier
// rejects ``forge.tensor.broadcast_add`` outright, so the fuse pass's
// rewrite is never attempted.
//
// FileCheck expectations:
//   * a non-zero exit code (the ``not`` wrapper flips that to a
//     passing check)
//   * stderr contains the canonical verifier substring

// CHECK-ERR: operand shapes are not broadcastable
func.func @bad_fuse_input(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<32x32xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %1 = forge.tensor.broadcast_add %0, %b
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<32x32xf32>) -> !forge.tensor<2x512x11008xf32>
    %2 = forge.tensor.fused_epilogue %1 {activation = #forge.activation<gelu>}
        : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %2 : !forge.tensor<2x512x11008xf32>
}
