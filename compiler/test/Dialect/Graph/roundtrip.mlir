// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s
// CHECK-LABEL: func.func @fused_mlp_graph
func.func @fused_mlp_graph(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %0 = forge.graph.matmul %x, %w : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.graph.matmul
    // CHECK-SAME: !forge.tensor<2x512x4096xf32>
    // CHECK-SAME: !forge.tensor<4096x11008xf32>
    // CHECK-SAME: -> !forge.tensor<2x512x11008xf32>
    %1 = forge.graph.add %0, %b : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.graph.add
    // CHECK-SAME: !forge.tensor<2x512x11008xf32>
    // CHECK-SAME: !forge.tensor<11008xf32>
    // CHECK-SAME: -> !forge.tensor<2x512x11008xf32>
    %2 = forge.graph.gelu %1 {approximate = "none"} : (!forge.tensor<2x512x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    // CHECK: forge.graph.gelu
    // CHECK-SAME: approximate = "none"
    // CHECK-SAME: !forge.tensor<2x512x11008xf32>
    // CHECK-SAME: -> !forge.tensor<2x512x11008xf32>
    // CHECK: return {{.*}} : !forge.tensor<2x512x11008xf32>
    return %2 : !forge.tensor<2x512x11008xf32>
}
