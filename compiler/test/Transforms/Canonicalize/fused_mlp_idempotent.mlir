// RUN: %forge-opt %s --pass-pipeline='canonicalize' | %FileCheck %s

// Idempotency golden test for the canonicalize pass on the FusedMLP
// tensor-IR triple (gemm + broadcast_add + fused_epilogue). The pass is
// wired with four RewritePattern stubs — dead-code elimination, common-
// subexpression elimination, constant folding, and fused-epilogue attribute
// canonicalisation — none of which fire on this well-formed triple, so the
// output is structurally identical to the input.
//
// FileCheck expectations:
//   * function name preserved (@fused_mlp_canonical)
//   * the three tensor ops survive verbatim
//   * no `forge.tensor.gelu` appears (the FusedMLP triple uses the fused
//     epilogue for activation; a `gelu` here would mean the pass rewrote
//     the fused_epilogue into a stand-alone gelu, which we explicitly do
//     NOT do in Phase 2)

// CHECK-LABEL: func.func @fused_mlp_canonical
func.func @fused_mlp_canonical(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.gemm
    // CHECK-SAME: !forge.tensor<2x512x4096xf32>
    %1 = forge.tensor.broadcast_add %0, %b
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.broadcast_add
    // CHECK-SAME: !forge.tensor<2x512x11008xf32>
    %2 = forge.tensor.fused_epilogue %1, %b {activation = #forge.activation<gelu>}
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.fused_epilogue
    // CHECK-SAME: activation = #forge.activation<gelu>
    // CHECK-NOT: forge.tensor.gelu
    // CHECK: {{func.return|return}}
    return %2 : !forge.tensor<2x512x11008xf32>
}
