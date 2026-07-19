# compiler/lib/Transforms/Canonicalize/__init__.py
#
# `canonicalize` xDSL module pass — Phase 2 / Task 4.
#
# Driver-side name: ``canonicalize``. Used by
# ``compiler/tools/forge-opt/forge_opt.py`` via
# ``--pass-pipeline='canonicalize,...'`` (Task 5).
#
# This pass wires four ``RewritePattern`` stubs that collectively make up
# the canonicalisation skeleton used by every MLIR-style transform stack:
#
# 1. **Dead-code elimination (DCE).** Removes ops whose results have no
#    uses AND whose traits guarantee no observable side effect. Phase 2
#    scopes DCE to a single safe subset: ops that carry BOTH the ``Pure``
#    AND ``NoMemoryEffect`` trait (which every Phase 1 ``forge.tensor.*``
#    op carries), AND whose ``out`` operand has zero consumers AND which
#    are not function/block-level terminators. The conservative subset
#    leaves the full DCE walker (xDSL's ``dead_code_elimination``) for
#    Phase 3, when we will also have an explicit memory-effect interface
#    to override the trait-based check.
#
# 2. **Common-subexpression elimination (CSE).** STUB. Equal-SSA-value
#    tracking requires a fresh pass-managed hash-cons map; Phase 2
#    registers the empty pattern (so the applier shape is consistent and
#    downstream passes can rely on the canonicalise entry-point producing
#    a deterministic post-order) but does not perform any rewrites yet.
#    CSE will land once Phase 3's expression-level hashing is in place.
#
# 3. **Constant folding.** STUB. ``forge.tensor.broadcast_add(%x, c0)``
#    would be the obvious target (replace the add with %x) but the
#    ``forge.tensor.constant`` op is not part of Phase 2's scope — it
#    belongs to Phase 3 (ForgeRT memory planning). The pattern exists as
#    a recognised "no-op until future work" hook.
#
# 4. **Attribute canonicalisation.** STUB. ``forge.tensor.fused_epilogue``
#    with ``activation = #forge.activation<none>`` and no bias is
#    semantically an identity (a no-op elementwise function); Phase 2
#    registers the pattern but, lacking an identity op (``reshape`` ships
#    in Phase 3, see
#    ``compiler/lib/Conversion/GraphToTensor/__init__.py`` §ViewPassThroughPattern),
#    leaves the op in place rather than rewriting it to something that
#    would still have to be peeled back off later.
#
# Idempotency: with all four patterns in their non-firing default state,
# feeding already-canonical IR through the pass produces bit-identical
# output. Exercised explicitly by
# ``compiler/test/Transforms/Canonicalize/fused_mlp_idempotent.mlir``.

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
from xdsl.traits import NoMemoryEffect, Pure

from compiler.lib.Dialect.Tensor import FusedEpilogueOp, GeluOp

# ---------------------------------------------------------------------------
# DCE pattern
# ---------------------------------------------------------------------------


class DeadGeluPattern(RewritePattern):
    """Erase unused ``forge.tensor.gelu`` ops.

    Conservative Phase 2 DCE: an op is "trivially dead" iff (a) every
    result has zero users, (b) the op carries both the ``Pure`` and
    ``NoMemoryEffect`` traits, and (c) it is not a terminator. The check
    is restricted to ``forge.tensor.gelu`` because GELU is the only
    elementwise op where we have an end-to-end aliasing-free guarantee
    at this stage; widening the DCE set happens in Phase 3 once the
    full memory-effect interface lands.

    When the match fires, ``rewriter.erase_op`` performs the structural
    erase; the rewriter verifies that no SSA uses remain (which our
    match already ruled out).
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, GeluOp):
            return
        if not op.has_trait(Pure, value_if_unregistered=False):
            return
        if not op.has_trait(NoMemoryEffect, value_if_unregistered=False):
            return
        # Trivially dead: every result unused.
        if any(result.first_use is not None for result in op.results):
            return
        rewriter.erase_op(op)


# ---------------------------------------------------------------------------
# CSE pattern (Phase 2 stub)
# ---------------------------------------------------------------------------


class CseStubPattern(RewritePattern):
    """Common-subexpression elimination — Phase 2 STUB.

    Equal-SSA-value tracking requires a hash-cons map keyed on
    (op-name, type, input values); building that map lives in Phase 3
    alongside the expression-level hashing that the constant folder
    will share. The stub exists in the applier so the canonicalisation
    entry point has the structurally-complete pattern list that future
    passes will rely on.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        return  # never matches in Phase 2


# ---------------------------------------------------------------------------
# Constant-folding pattern (Phase 2 stub)
# ---------------------------------------------------------------------------


class ConstFoldStubPattern(RewritePattern):
    """Constant folding — Phase 2 STUB.

    Phase 2 has no ``forge.tensor.constant`` op (that belongs to Phase 3's
    memory-planning work, where the ForgeRT provider can ingest pre-
    tiled constant pools). Constant-folding rewrites like
    ``broadcast_add(%x, c0)`` → ``%x`` therefore have nothing to match on.
    The stub is present so the applier signature stabilises early.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        return  # never matches in Phase 2


# ---------------------------------------------------------------------------
# Attribute canonicalisation pattern (Phase 2 stub)
# ---------------------------------------------------------------------------


class FusedEpilogueAttrCanonStubPattern(RewritePattern):
    """``fused_epilogue`` attribute canonicalisation — Phase 2 STUB.

    The fully-canonical form of
    ``forge.tensor.fused_epilogue %x {activation = #forge.activation<none>}``
    (with no bias) is the elementwise identity. Replacing it with an
    identity op requires either ``forge.tensor.reshape`` (which ships
    in Phase 3 once the GraphToTensor pass lowers ``forge.graph.view``)
    or a brand-new ``forge.tensor.identity`` op. Phase 2 keeps the
    ``fused_epilogue`` op in place to avoid committing to either
    direction before the identity-op decision is finalised.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, FusedEpilogueOp):
            return
        # No rewrite yet — see module-level docstring.
        return


# ---------------------------------------------------------------------------
# Module pass
# ---------------------------------------------------------------------------


class CanonicalizePass(ModulePass):
    """Apply the Phase 2 canonicalisation pattern set to a module.

    The pass name ``canonicalize`` matches the value the CLI driver
    expects in ``--pass-pipeline='canonicalize'`` (see
    ``compiler/tools/forge-opt/forge_opt.py``, Task 5). It runs a
    ``PatternRewriteWalker`` with the four stub patterns above in their
    declared order (DCE → CSE → const-fold → attr-canon).

    ``apply_recursively=False`` matches the GraphToTensor pass: the
    pattern set only matches top-level ops, and the recursive walk would
    re-traverse regions we already know will never match. The flag is
    revisited in Phase 3 once the pattern set grows region-aware
    variants.
    """

    name = "canonicalize"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    DeadGeluPattern(),
                    CseStubPattern(),
                    ConstFoldStubPattern(),
                    FusedEpilogueAttrCanonStubPattern(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
