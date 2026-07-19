# compiler/lib/Transforms/Fuse/__init__.py
#
# `fuse` xDSL module pass — Phase 2 / Task 7.
#
# Driver-side name: ``fuse``. Invoked via ``--pass-pipeline='fuse'``
# (Task 5) or as part of ``--action=optimize``'s default chain.
#
# Phase 2 collapses the tail of the canonical FusedMLP triple:
#
#     %0 = forge.tensor.gemm %a, %w {…}
#     %1 = forge.tensor.broadcast_add %0, %b
#     %2 = forge.tensor.fused_epilogue %1 {activation = gelu|relu}
#
# into
#
#     %0 = forge.tensor.gemm %a, %w {…}
#     %2' = forge.tensor.fused_epilogue %0, %b {activation = gelu|relu}
#
# i.e. the broadcast_add is elided and its second operand becomes the
# fused_epilogue's bias. The activation must be ``gelu`` or ``relu``;
# ``<none>`` / ``<silu>`` are left untouched.
#
# Why 3 → 2 (not 3 → 1 as the spec sketch suggested): the existing
# ``forge.tensor.fused_epilogue`` hard-pins ``x.shape == out.shape`` in
# its verifier, so the dialect cannot subsume a gemm that changes the
# trailing dimensions. A literal 3 → 1 collapse would need a new
# ``forge.tensor.fused_gemm_epilogue`` op — Phase 3 dialect work, out
# of scope for Task 7. The 3 → 2 collapse still removes a
# broadcast_add + bare fused_epilogue and produces the canonical
# FusedMLP IR.
#
# Idempotency: re-running the pass on the post-fusion IR sees the new
# fused_epilogue has a bias (so the bare-fused_epilogue match fails)
# and no preceding broadcast_add, so the pattern does not fire.

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compiler.lib.Dialect.Tensor import (
    Activation,
    BroadcastAddOp,
    FusedEpilogueOp,
    GemmOp,
)


class FuseAddActivationPattern(RewritePattern):
    """Coalesce a ``broadcast_add → bare fused_epilogue`` pair."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, FusedEpilogueOp):
            return
        if op.activation.data not in (Activation.GELU, Activation.RELU):
            return
        if op.bias is not None:
            return

        if not isinstance(op.x.owner, BroadcastAddOp):
            return
        add = op.x.owner

        if not isinstance(add.x.owner, GemmOp):
            return
        gemm = add.x.owner
        gemm_users = [use.operation for use in gemm.out.uses]
        if len(gemm_users) != 1 or gemm_users[0] is not add:
            return

        new_op = FusedEpilogueOp.create(
            operands=[gemm.out, add.y],
            result_types=[op.out.type],
            properties={"activation": op.activation},
        )
        # Order matters: replace the tail first so ``add.out`` is
        # unwired from the live IR (its only remaining use was
        # ``op.x``); only then can we erase ``add``. ``gemm`` stays —
        # its result now feeds the new op.
        rewriter.replace_op(op, new_op)
        rewriter.erase_op(add)


class FusePass(ModulePass):
    """Apply the Phase 2 fusion pattern to a module.

    Pass name ``fuse`` matches the CLI driver's ``--pass-pipeline='fuse'``
    value (Task 5). ``apply_recursively=False`` — the pattern only
    matches the add-then-bare-activation pair at one level.
    """

    name = "fuse"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FuseAddActivationPattern(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
