// RUN: not %forge-opt %s --action=lower-astra 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test for the TensorToAstra conversion pass (Wave 9 / Task 42):
// the post-walk audit rejects any `forge.tensor.*` op that has no astra
// equivalent. We use `forge.tensor.broadcast_add` in a tail position (no
// downstream matmul to fold the bias into) — the
// :class:`BroadcastAddPattern` deliberately leaves tail-position
// broadcast_add ops alone, so the audit sees them as leftovers and raises
// the canonical "no astra equivalent" error.
//
// FileCheck expectations:
//   * process exits non-zero (the `not` wrapper flips that)
//   * stderr contains the canonical "no astra equivalent" substring

func.func @unsupported_op(
    %x: !forge.tensor<2x512x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    // CHECK-ERR: TensorToAstra is not part of the shipped pipeline
    %z = forge.tensor.broadcast_add %x, %b
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %z : !forge.tensor<2x512x11008xf32>
}
