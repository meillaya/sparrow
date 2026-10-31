"""In-process lowering pipeline used by the frontend plan builder."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from compiler.lib.context import build_context
from compiler.lib.Conversion.GraphToTensor import GraphToTensorPass
from compiler.lib.Transforms.Canonicalize import CanonicalizePass
from compiler.lib.Transforms.Fuse import FusePass
from compiler.lib.Transforms.ShapeInference import ShapeInferencePass


def lower_to_tensor(module: ModuleOp, context: Context | None = None) -> ModuleOp:
    """Lower graph ops to tensor ops and run the local optimization chain.

    The order is graph-to-tensor, shape-inference, fuse, canonicalize. The
    module is verified after every pass so a broken rewrite fails at the
    pass that produced it.
    """
    ctx = context if context is not None else build_context()
    for pass_ in (
        GraphToTensorPass(),
        ShapeInferencePass(),
        FusePass(),
        CanonicalizePass(),
    ):
        pass_.apply(ctx, module)
        module.verify()
    return module


__all__ = ("lower_to_tensor",)
