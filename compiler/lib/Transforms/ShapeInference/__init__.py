# compiler/lib/Transforms/ShapeInference/__init__.py
#
# `shape-inference` xDSL module pass — Phase 2 / Task 6.
#
# Driver-side name: ``shape-inference``. Invoked via
# ``--pass-pipeline='shape-inference'`` (Task 5) or as part of
# ``--action=optimize``'s default chain.
#
# The pass walks every op in a module and rewrites the op's result type
# to whatever the pass computes from the operand shapes and (when
# relevant) the transpose / activation properties. Symbolic dimensions
# (``StringAttr``) propagate unchanged; static dimensions come from the
# op's deterministic shape rules below.
#
# Per-op rules (spec §5):
#
#   * ``forge.tensor.gemm(A, B)``
#       → result shape = (batch(A) + [M from a, N from b])
#       with the inner-K agreement ``A.shape[-1] == B.shape[-2]``
#       enforced by the dialect verifier.
#
#   * ``forge.tensor.broadcast_add(A, B)``
#       → result shape = NumPy-broadcast(A.shape, B.shape)
#       (already exposed by ``compiler.lib.Dialect.Tensor._broadcast_shape``).
#
#   * ``forge.tensor.gelu(X)``
#       → result shape = X.shape
#
#   * ``forge.tensor.fused_epilogue(X, B)``
#       → result shape = X.shape
#
# Symbolic dims (``StringAttr``) flow through every rule unchanged, so
# dynamic-shape IR leaves the pass identically. The pass is therefore
# idempotent: re-running it on its own output is a structural no-op.

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compiler.lib.Dialect.Graph import TensorType
from compiler.lib.Dialect.Graph import _broadcast_shapes as _graph_broadcast_shapes
from compiler.lib.Dialect.Tensor import (
    BroadcastAddOp,
    FusedEpilogueOp,
    GeluOp,
    GemmOp,
)

# ---------------------------------------------------------------------------
# Pure helpers — exposed at module level so lit goldens that want to
# construct shape rules manually (e.g. cross-checking in a future
# dialect verifier test) can reach them without reaching into private
# tensor-dialect names.
# ---------------------------------------------------------------------------


def _make_tensor_type(shape: Sequence[Attribute], dtype) -> TensorType:
    """Build a fresh ``TensorType`` from a (possibly symbolic) shape tuple.

    The runtime type of each ``shape`` element is ``IntAttr`` or
    ``StringAttr`` (the ``TensorType`` verifier rule), but the static
    type tuple we hold at the call site is the wider
    ``tuple[Attribute, ...]`` — we cast through the canonical
    ``IntAttr``/``StringAttr`` enum so ty accepts the call.
    """
    from xdsl.dialects.builtin import IntAttr, StringAttr

    coerced: list[IntAttr | StringAttr] = []
    for dim in shape:
        if isinstance(dim, (IntAttr, StringAttr)):
            coerced.append(dim)
        else:  # pragma: no cover — TensorType verifier rejects other dim kinds
            raise TypeError(f"unexpected dim attribute type: {type(dim).__name__}")
    return TensorType(coerced, dtype)


def _gemm_out_shape(a_type: TensorType, b_type: TensorType) -> tuple[Attribute, ...]:
    """``[A.shape[:-1], B.shape[-1]]`` (the dialect verifier already
    confirmed ``A.shape[-1] == B.shape[-2]``)."""
    return (*a_type.get_shape()[:-1], b_type.get_shape()[-1])


def _broadcast_op_shape(a_type: TensorType, b_type: TensorType) -> tuple[Attribute, ...] | None:
    """NumPy broadcast of two tensor shapes (static + symbolic dims).

    Defers to ``compiler.lib.Dialect.Graph._broadcast_shapes`` which is
    the dialect's canonical mixed-static/symbolic broadcast helper;
    returning ``None`` on incompatibility so the caller can short-circuit.
    """
    return _graph_broadcast_shapes(a_type.get_shape(), b_type.get_shape())


# ---------------------------------------------------------------------------
# Per-op pattern
# ---------------------------------------------------------------------------


class ShapeInferencePattern(RewritePattern):
    """``RewritePattern`` rewriting each ``forge.tensor.*`` op's result
    type to the shape the pass computes.

    The match logic per op-class:

    * ``forge.tensor.gemm`` → ``[batch(A), N]``
    * ``forge.tensor.broadcast_add`` → ``broadcast(A.shape, B.shape)``
    * ``forge.tensor.gelu`` → ``X.shape`` (1-result)
    * ``forge.tensor.fused_epilogue`` → ``X.shape`` (1-result)

    The rewrite only fires when the computed type differs from the
    existing one (cheap fast-path); ``rewriter.replace_value_with_new_type``
    cascades the type change through downstream SSA uses.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if isinstance(op, GemmOp):
            a_t = _as_tensor_type(op.a.type)
            b_t = _as_tensor_type(op.b.type)
            new_shape = _gemm_out_shape(a_t, b_t)
            new_type = _make_tensor_type(new_shape, a_t.get_dtype())
            self._maybe_replace(op.out, new_type, rewriter)
            return
        if isinstance(op, BroadcastAddOp):
            x_t = _as_tensor_type(op.x.type)
            y_t = _as_tensor_type(op.y.type)
            new_shape = _broadcast_op_shape(x_t, y_t)
            if new_shape is None:
                return
            new_type = _make_tensor_type(new_shape, x_t.get_dtype())
            self._maybe_replace(op.out, new_type, rewriter)
            return
        if isinstance(op, GeluOp):
            x_t = _as_tensor_type(op.x.type)
            self._maybe_replace(
                op.out, _make_tensor_type(x_t.get_shape(), x_t.get_dtype()), rewriter
            )
            return
        if isinstance(op, FusedEpilogueOp):
            x_t = _as_tensor_type(op.x.type)
            self._maybe_replace(
                op.out, _make_tensor_type(x_t.get_shape(), x_t.get_dtype()), rewriter
            )
            return

    @staticmethod
    def _maybe_replace(old_result, new_type: TensorType, rewriter: PatternRewriter) -> None:
        if old_result.type == new_type:
            return
        rewriter.replace_value_with_new_type(old_result, new_type)


def _as_tensor_type(type_attr: Attribute) -> TensorType:
    """Narrow ``Attribute`` to ``TensorType`` (defensive — dialect
    verifier already guarantees this at every join point)."""
    return cast(TensorType, type_attr)


# ---------------------------------------------------------------------------
# Module pass
# ---------------------------------------------------------------------------


class ShapeInferencePass(ModulePass):
    """Run shape inference over every op in a module.

    The pass name ``shape-inference`` matches the value the CLI driver
    expects in ``--pass-pipeline='shape-inference'`` (Task 5). Applies a
    single ``ShapeInferencePattern`` via the ``PatternRewriteWalker``,
    which is sufficient because the pattern operates orthogonally per op
    (no cross-op rewrites, no convergence concerns).

    ``apply_recursively=False`` is fine: the dialect guarantees every
    ``forge.tensor.*`` op's operands are nested only through block
    arguments (which the walker sees at the same level).
    """

    name = "shape-inference"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([ShapeInferencePattern()]),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
