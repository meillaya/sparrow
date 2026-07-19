// RUN: %forge-opt %s --pass-pipeline='shape-inference' | %FileCheck %s

// Golden test for the Phase 2 shape-inference pass on a fully-static
// FusedMLP-style IR. Each result type is replaced by the shape the pass
// computes from inputs + attributes, so for fully-static inputs the
// output is bit-identical to the input — this test exercises the
// "replace-with-same-type" no-op rewrite path and confirms the verifier
// still accepts the rewritten module.
//
// The pass also handles symbolic-dim IR (StringAttr dims) by passing
// them through unchanged; that path is exercised by the in-tree Python
// smoke test in `compiler/lib/Transforms/ShapeInference/__init__.py`
// because the Phase 1 `forge.tensor` MLIR parser does not yet support
// symbolic dim names whose trailing tokens happen to look like dtype
// spellings (a Phase 3 parser fix).
//
// FileCheck expectations:
//   * function name preserved (@shape_inference_static)
//   * result types are exactly the broadcast / matmul-output shapes

// CHECK-LABEL: func.func @shape_inference_static
func.func @shape_inference_static(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    // CHECK: forge.tensor.gemm
    // CHECK-SAME: -> !forge.tensor<2x512x11008xf32>
    %0 = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.broadcast_add
    // CHECK-SAME: -> !forge.tensor<2x512x11008xf32>
    %1 = forge.tensor.broadcast_add %0, %b
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.tensor.gelu
    // CHECK-SAME: -> !forge.tensor<2x512x11008xf32>
    %2 = forge.tensor.gelu %1
        : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: {{func.return|return}}
    return %2 : !forge.tensor<2x512x11008xf32>
}

