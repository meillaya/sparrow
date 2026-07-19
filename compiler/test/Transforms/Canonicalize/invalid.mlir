// RUN: not %forge-opt %s --pass-pipeline='canonicalize' 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test: an IR fragment whose `forge.tensor.broadcast_add`
// operands have non-broadcastable shapes must be rejected by the
// `forge.tensor` verifier BEFORE the canonicalize pass runs. The exact
// substring (`operand shapes are not broadcastable`) is emitted by the
// tensor dialect's verifier in `compiler/lib/Dialect/Tensor/__init__.py`.
//
// FileCheck expectations:
//   * process exits non-zero (the `not` wrapper flips that)
//   * stderr contains the canonical error substring

func.func @bad_canonicalize_input(
    %x: !forge.tensor<2x3xf32>,
    %y: !forge.tensor<4x5xf32>
) -> !forge.tensor<2x3xf32> {
    // CHECK-ERR: operand shapes are not broadcastable
    %0 = forge.tensor.broadcast_add %x, %y
        : (!forge.tensor<2x3xf32>, !forge.tensor<4x5xf32>) -> !forge.tensor<2x3xf32>
    return %0 : !forge.tensor<2x3xf32>
}
