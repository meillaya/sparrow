// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s
//
// Named symbolic dimensions are lexed glued to the ``x`` shape separator
// (``Bx4xf32`` arrives as a single BARE_IDENT), so these tests pin the
// parser's split-and-resume behavior. Symbolic dimension names must not
// contain the letter ``x``.

// CHECK-LABEL: func.func @named_symbol_dim
// CHECK-SAME: !forge.tensor<Bx4xf32>
// CHECK: func.return {{.*}} : !forge.tensor<Bx4xf32>
func.func @named_symbol_dim(%x: !forge.tensor<Bx4xf32>) -> !forge.tensor<Bx4xf32> {
    return %x : !forge.tensor<Bx4xf32>
}

// CHECK-LABEL: func.func @two_named_symbol_dims
// CHECK-SAME: !forge.tensor<BxSx4096xf32>
// CHECK: func.return {{.*}} : !forge.tensor<BxSx4096xf32>
func.func @two_named_symbol_dims(%x: !forge.tensor<BxSx4096xf32>) -> !forge.tensor<BxSx4096xf32> {
    return %x : !forge.tensor<BxSx4096xf32>
}

// CHECK-LABEL: func.func @anonymous_symbols_with_constraints
// CHECK-SAME: !forge.tensor<?x?x4096xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
// CHECK: func.return {{.*}} : !forge.tensor<?x?x4096xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
func.func @anonymous_symbols_with_constraints(
    %x: !forge.tensor<?x?x4096xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
) -> !forge.tensor<?x?x4096xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]> {
    return %x : !forge.tensor<?x?x4096xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
}
