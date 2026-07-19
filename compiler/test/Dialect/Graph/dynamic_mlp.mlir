// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s
// CHECK-LABEL: func.func @dynamic_mlp
func.func @dynamic_mlp(
    %x: !forge.tensor<?x?x4096xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]> {
    %0 = forge.graph.matmul %x, %w : (!forge.tensor<?x?x4096xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
    // CHECK: forge.graph.matmul
    // CHECK-SAME: !forge.tensor<?x?x4096xf32
    // CHECK-SAME: !forge.tensor<4096x11008xf32>
    // CHECK-SAME: -> !forge.tensor<?x?x11008xf32
    %1 = forge.graph.add %0, %b : (!forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>, !forge.tensor<11008xf32>) -> !forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
    // CHECK: forge.graph.add
    %2 = forge.graph.symbolic_guard %1 {constraint = #forge.symbolic_constraint<"K", equal, 4096>} : (!forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>) -> !forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
    // CHECK: forge.graph.symbolic_guard
    // CHECK-SAME: constraint = #forge.symbolic_constraint<"K", equal, 4096>
    return %2 : !forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
}