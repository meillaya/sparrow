# compiler/lib/Conversion/GraphToTensor/__init__.py
#
# `forge.graph` → `forge.tensor` conversion pass (Task 7).
#
# Lowering strategy: every `forge.graph.<op>` is rewritten 1:1 into its
# `forge.tensor` equivalent. We deliberately do NOT do any cross-op
# fusion here — the spec (§5) describes `forge.tensor.fusion { ... }`
# regions as the home for `gemm + bias + activation` style coalescing,
# which would be a separate pass operating on the lowered IR.
#
# This module wires four xDSL `RewritePattern`s (one per supported graph op)
# into a `GreedyRewritePatternApplier` and drives them through a
# `PatternRewriteWalker`, mounted on a `GraphToTensorPass` `ModulePass`.
#
# Per-op mapping (Task 7 spec):
#
#     forge.graph.matmul %x, %w
#         → forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
#
#     forge.graph.add %a, %b
#         → forge.tensor.broadcast_add %a, %b
#
#     forge.graph.gelu %x {approximate = "none"|"tanh"|absent}
#         → forge.tensor.gelu %x {approximate = true}        (when "tanh")
#         → forge.tensor.gelu %x                              (when "none" / absent)
#         (The tensor dialect models the attr as a `BoolAttr` where `true`
#         selects the tanh approximation; "none" matches the default `false`
#         and elides the property entirely via `ParsePropInAttrDict`.)
#
#     forge.graph.view %x
#         → left untouched (the FusedMLP vertical slice does not use
#         `view`; emitting `forge.tensor.reshape` would require a new
#         op + verification, which is not part of the shipped scope.)
#
# Idempotency: no `RewritePattern` matches a `forge.tensor.*` op, so the pass
# is a no-op when fed already-lowered IR. This is exercised explicitly by
# `compiler/test/Conversion/GraphToTensor/idempotent.mlir`.
#
# See the phase 7 notes for the gotchas hit
# while wiring this up — most notably the StringAttr↔BoolAttr mismatch on
# the `approximate` property and the assembler-format quirk where `attr-dict`
# silently drops default-valued `BoolAttr` properties.

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import BoolAttr, ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.lib.Dialect.Graph import (
    AddOp,
    GeluApproxAttr,
    GeluOp,
    MatmulOp,
    StringAttr,
    ViewOp,
)
from compiler.lib.Dialect.Tensor import (
    BroadcastAddOp,
    GemmOp,
)
from compiler.lib.Dialect.Tensor import (
    GeluOp as TensorGeluOp,
)

# ---------------------------------------------------------------------------
# Per-op rewrite patterns
# ---------------------------------------------------------------------------


class MatmulToGemmPattern(RewritePattern):
    """Rewrite ``forge.graph.matmul`` → ``forge.tensor.gemm``.

    Phase 1 keeps the transpose flags at their default ``false`` values —
    pre-transposed inputs land in ``forge.graph`` with the contract dims
    already swapped (since FX captures the user-visible PyTorch layout).
    A future pass can fold transposes into the operands instead of
    encoding them as flags; spec §5 keeps the door open.

    The result type is read off the source op; both ops carry an explicit
    ``!forge.tensor<...>`` shape on their single result, and the
    ``forge.tensor.gemm`` verifier will re-check the inner-K agreement
    against the operands.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        new_op = GemmOp.create(
            operands=[op.x, op.w],
            result_types=[op.out.type],
            properties={
                "transpose_a": BoolAttr.from_bool(False),
                "transpose_b": BoolAttr.from_bool(False),
            },
        )
        rewriter.replace_op(op, new_op)


class AddToBroadcastAddPattern(RewritePattern):
    """Rewrite ``forge.graph.add`` → ``forge.tensor.broadcast_add``.

    NumPy broadcasting is identical between the two ops — the graph
    dialect just bakes in autograd semantics that the tensor dialect
    elides. Operands and result type pass through unchanged.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter, /) -> None:
        new_op = BroadcastAddOp.create(
            operands=[op.a, op.b],
            result_types=[op.out.type],
        )
        rewriter.replace_op(op, new_op)


class GeluToGeluPattern(RewritePattern):
    """Rewrite ``forge.graph.gelu`` → ``forge.tensor.gelu``.

    The graph dialect carries ``approximate`` as a ``StringAttr``
    (``"none"`` = exact erf, ``"tanh"`` = tanh approximation); the tensor
    dialect models the same flag as a ``BoolAttr`` whose default ``false``
    means "exact". We therefore:

    * Translate ``"tanh"`` → ``approximate = true``.
    * Drop ``"none"`` (or absent) entirely so the property elides from
      the printed IR via ``ParsePropInAttrDict`` (see the dialect source
      — the tensor dialect's GeluOp is declared with
      ``irdl_options = (ParsePropInAttrDict(),)``).

    **xDSL quirk:** the graph dialect declares
    ``approximate = prop_def(StringAttr, default_value=StringAttr(...))``,
    and the assembly format includes both an optional ``(approximate = $approximate)``
    group AND a trailing ``attr-dict``. Because the prop has a
    ``default_value``, the property accessor ``op.approximate`` always
    returns the default — the user-supplied value lands in
    ``op.attributes["approximate"]`` instead. We read from the attribute
    dict first and fall back to the property (handles both the explicit
    attr-dict parse path and the optional-group parse path).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GeluOp, rewriter: PatternRewriter, /) -> None:
        approx_attr = op.attributes.get("approximate")
        if approx_attr is not None and isinstance(approx_attr, GeluApproxAttr | StringAttr):  # type: ignore[misc]
            approx_value = approx_attr.data
        else:
            approx_value = op.approximate.data  # type: ignore[attr-defined]
        if approx_value == "tanh":
            properties: dict[str, object] = {"approximate": BoolAttr.from_bool(True)}
        else:
            properties = {}
        new_op = TensorGeluOp.create(
            operands=[op.x],
            result_types=[op.out.type],
            properties=properties,
        )
        rewriter.replace_op(op, new_op)


class ViewPassThroughPattern(RewritePattern):
    """Leave ``forge.graph.view`` alone in the shipped scope.

    The FusedMLP vertical slice never emits ``view``; lowering it would
    require either a brand-new ``forge.tensor.reshape`` op (with its own
    verifier and shape-inference story) or folding the reshape into a
    downstream ``fused_epilogue``. The shipped pass keeps the op
    untouched. This pattern is intentionally empty — its presence in the
    applier signals intent (we considered the op and chose not to lower
    it) and keeps the applier shape consistent.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ViewOp, rewriter: PatternRewriter, /) -> None:
        return


# ---------------------------------------------------------------------------
# Module pass
# ---------------------------------------------------------------------------


class GraphToTensorPass(ModulePass):
    """Lower a module's ``forge.graph.*`` ops into ``forge.tensor.*`` ops.

    The pass name ``graph-to-tensor`` matches the value the CLI driver
    expects in ``--pass-pipeline='builtin.module(graph-to-tensor)'`` (see
    `compiler/tools/forge-opt/forge_opt.py`). It expects both dialects to
    be loaded into the context — the driver does this via
    ``discover_dialects()`` so callers do not need to register them
    manually.
    """

    name = "graph-to-tensor"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MatmulToGemmPattern(),
                    AddToBroadcastAddPattern(),
                    GeluToGeluPattern(),
                    ViewPassThroughPattern(),
                ]
            ),
            # `apply_recursively=False` is the conservative choice for a
            # 1:1 dialect lowering: each op rewrite is independent of the
            # others, and we don't want the walker to re-visit the new
            # tensor op after the rewrite (none of the patterns match
            # anyway, but skipping the re-traversal keeps the debug IR
            # clean and avoids the double-walk the recursive variant
            # triggers on nested regions).
            apply_recursively=False,
        )
        walker.rewrite_module(op)
