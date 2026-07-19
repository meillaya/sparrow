// RUN: not %forge-opt %s --action=lower 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test: a malformed matmul (inner K dim mismatch) must be
// rejected by the graph dialect's verifier BEFORE the conversion pass
// even gets a chance to run. The exact text of the error message
// (`matmul inner dims mismatch`) matches the substring emitted by
// `forge.graph.matmul`'s `verify_` hook in `compiler/lib/Dialect/Graph/__init__.py`.
//
// FileCheck expectations:
//   * process exits non-zero (the `not` wrapper flips that)
//   * stderr contains the canonical error substring

func.func @bad_matmul_input(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<2048x11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    // CHECK-ERR: matmul inner dims mismatch
    %0 = forge.graph.matmul %x, %w : (!forge.tensor<2x512x4096xf32>, !forge.tensor<2048x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %0 : !forge.tensor<2x512x11008xf32>
}