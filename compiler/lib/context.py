"""Shared xDSL context for the compiler and the plan builder."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func

from compiler.lib.Dialect.Astra import Astra
from compiler.lib.Dialect.Graph import Graph
from compiler.lib.Dialect.MicroGPU import MicroGPU
from compiler.lib.Dialect.Schedule import Schedule
from compiler.lib.Dialect.Tensor import Tensor


def build_context() -> Context:
    """Return a context with every Sparrow dialect loaded.

    The frontend uses this for in-process lowering. The forge-opt driver
    keeps its own discovery path for the CLI; both load the same dialects.
    """
    context = Context(allow_unregistered=False)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    for dialect in (Graph, Tensor, Schedule, Astra, MicroGPU):
        context.load_dialect(dialect)
    return context
