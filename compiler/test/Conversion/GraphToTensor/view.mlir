// RUN: %forge-opt %s --action=lower | %FileCheck %s

// ``forge.graph.view`` lowers to ``forge.tensor.reshape`` so the plan
// builder has an explicit tensor-dialect node for every graph op.

// CHECK-LABEL: func.func @view_roundtrip
// CHECK: forge.tensor.reshape
// CHECK-SAME: !forge.tensor<2x512x4096xf32>
// CHECK-SAME: -> !forge.tensor<2x2097152xf32>
// CHECK-NOT: forge.graph.view
func.func @view_roundtrip(%x: !forge.tensor<2x512x4096xf32>) -> !forge.tensor<2x2097152xf32> {
    %0 = forge.graph.view %x : (!forge.tensor<2x512x4096xf32>) -> !forge.tensor<2x2097152xf32>
    return %0 : !forge.tensor<2x2097152xf32>
}
