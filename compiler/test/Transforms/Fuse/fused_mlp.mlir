// RUN: %forge-opt %s --pass-pipeline='fuse' --action=optimize | %FileCheck %s

// Golden test for the Phase 2 fusion pass on the canonical FusedMLP
// triple. The input has three tensor ops in the body:
//
//     %0 = forge.tensor.gemm %x, %w {…}
//     %1 = forge.tensor.broadcast_add %0, %b
//     %2 = forge.tensor.fused_epilogue %1 {activation = gelu}
//
// The fused_epilogue (last in the triple) has no bias of its own: the
// dialect verifier hard-pins ``x.shape == out.shape`` on
// ``forge.tensor.fused_epilogue``, so a Phase-2 combine that pulled
// the ``gemm`` into a single op would require a NEW
// ``forge.tensor.fused_gemm_epilogue`` op (Phase 3 territory, not
// part of the Phase 2 dialect surface). What Phase 2 *can* do is
// collapse the (broadcast_add → bare fused_epilogue) tail into a
// single bias-bearing fused_epilogue, producing the 2-op output
//
//     %0 = forge.tensor.gemm %x, %w {…}            (untouched)
//     %1' = forge.tensor.fused_epilogue %0, %b {activation = gelu}
//
// …which is functionally equivalent to ``gelu(gemm(x, w) + b)`` —
// the canonical FusedMLP body.
//
// FileCheck expectations:
//   * the gemm op survives (no dialect surface to subsume it)
//   * the broadcast_add op is GONE
//   * EXACTLY one fused_epilogue op remains (was 2: the bare
//     activation and the final result), with `!forge.tensor<11008xf32>`
//     bias + `#forge.activation<gelu>` activation

// CHECK-LABEL: func.func @fused_mlp_after_fusion
// CHECK: forge.tensor.gemm
// CHECK-NOT: forge.tensor.broadcast_add
// CHECK: forge.tensor.fused_epilogue {{.*}}activation = #forge.activation<gelu>{{.*}}!forge.tensor<11008xf32>{{.*}}-> !forge.tensor<2x512x11008xf32>
// CHECK: {{func.return|return}}
func.func @fused_mlp_after_fusion(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %1 = forge.tensor.broadcast_add %0, %b
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %2 = forge.tensor.fused_epilogue %1 {activation = #forge.activation<gelu>}
        : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %2 : !forge.tensor<2x512x11008xf32>
}
