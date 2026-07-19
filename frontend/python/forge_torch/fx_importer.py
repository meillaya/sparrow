"""FX → MLIR importer for ForgeStack.

The :class:`FxImporter` walks a captured ``torch.fx.GraphModule`` and emits
``forge.graph`` dialect MLIR as a Python string. It is the Phase-1 bridge
between PyTorch's symbolic-graph capture (spec §3, item 3) and the xDSL
dialect surface defined under :mod:`compiler.lib.Dialect.Graph`.

Usage
-----
>>> from forge_torch.fx_importer import FxImporter
>>> imported = FxImporter(example_inputs).import_graph(gm)
>>> # ``imported`` is a textual MLIR module parseable by ``forge-opt --action=roundtrip``.

Design
------
The importer consumes the FX graph in two passes:

1. **Type inference pass** — for every node we compute the resulting
   ``!forge.tensor<...>`` shape + dtype, sourced from
   ``node.meta['tensor_meta']`` when present (the common path with
   ``make_fx`` / ``torch.export``) or from Python-level shape inference
   driven by :attr:`example_inputs` otherwise.
2. **Emission pass** — we build a ``func.func`` holding a single block
   whose argument types match the placeholders, then insert each op in
   FX order, resolving operands via a node-name → SSAValue map.

Op translation rules live in :mod:`forge_torch.op_mapping` — this module
only orchestrates the xDSL IRBuilder.
"""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path
from typing import ClassVar

import torch
from torch.fx import GraphModule, Node

# Prepend repo root so we can ``import compiler.lib.Dialect.*`` (a sibling of
# ``frontend/python/``). Mirrors compiler/tools/forge-opt/forge_opt.py.
_compiler_root = Path(__file__).resolve().parents[3]
_compiler_root_str = str(_compiler_root)
if _compiler_root_str not in sys.path and (_compiler_root / "compiler").is_dir():
    sys.path.insert(0, _compiler_root_str)

from xdsl.builder import Builder, InsertPoint  # noqa: E402
from xdsl.context import Context  # noqa: E402
from xdsl.dialects.builtin import Builtin, ModuleOp  # noqa: E402
from xdsl.dialects.func import FuncOp, ReturnOp  # noqa: E402
from xdsl.ir import Block, Region, SSAValue  # noqa: E402
from xdsl.printer import Printer  # noqa: E402

from forge_torch.op_mapping import DTYPE_TO_FORGE, map_node_to_forge  # noqa: E402


def _default_function_name(gm: GraphModule) -> str:
    name = getattr(gm, "_graph_name", None) or gm.__class__.__name__
    clean = str(name).replace(".", "_").replace(" ", "_")
    return f"{clean}_graph"


class FxImporter:
    """Walk a ``torch.fx.GraphModule`` and emit ``forge.graph`` MLIR text."""

    _GRAPH_OPS: ClassVar[dict[str, str]] = {
        "forge.graph.matmul": "matmul",
        "forge.graph.add": "add",
        "forge.graph.gelu": "gelu",
        "forge.graph.view": "view",
    }

    def __init__(self, example_inputs: list[torch.Tensor] | None = None) -> None:
        self._example_inputs: list[torch.Tensor] = list(example_inputs or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def import_graph(self, gm: GraphModule) -> str:
        graph = gm.graph if hasattr(gm, "graph") else gm
        node_types: dict[str, object] = self._infer_node_types(graph)

        placeholder_names, placeholder_types = [], []
        for name, typ in node_types.items():
            placeholder_names.append(name)
            placeholder_types.append(typ)
        # ``_infer_node_types`` walks in FX order and writes placeholders
        # first; partition them deterministically.
        ordered_names: list[str] = []
        ordered_types: list = []
        for node in list(graph.nodes):
            if node.op == "placeholder":
                ordered_names.append(node.name)
                ordered_types.append(node_types[node.name])
        placeholder_names = ordered_names
        placeholder_types = ordered_types

        output_node = _output_node(graph)
        if output_node is None:
            output_arg_names: list[str] = []
            output_types: list = []
        else:
            flat_args = _flatten_output_args(output_node.args)  # type: ignore[arg-type]
            output_arg_names = [_arg_name(a) for a in flat_args]
            output_types = [node_types[name] for name in output_arg_names]

        block = Block(arg_types=placeholder_types)
        func = FuncOp.from_region(
            _default_function_name(gm),
            input_types=placeholder_types,
            return_types=output_types,
            region=Region(block),
        )

        ssa_values: dict[str, SSAValue] = {}
        for arg, name in zip(block.args, placeholder_names, strict=True):
            ssa_values[name] = arg

        builder = Builder(InsertPoint.at_start(block))
        for node in list(graph.nodes):
            if node.op in ("placeholder", "output", "call_module"):
                continue
            if node.op == "call_function":
                self._emit_call_function(builder, node, ssa_values)
                continue
            if node.op == "get_attr":
                continue
            raise AssertionError(f"unhandled FX node.op: {node.op!r}")  # pragma: no cover

        if output_node is not None:
            operands = [ssa_values[name] for name in output_arg_names]
            builder.insert(ReturnOp.create(operands=operands, result_types=[]))
        elif output_types:
            builder.insert(ReturnOp.create(operands=[], result_types=[]))

        func.verify()
        module = ModuleOp([func])
        module.verify()

        buf = io.StringIO()
        Printer(stream=buf).print_op(module)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Type inference (pass 1)
    # ------------------------------------------------------------------

    def _infer_node_types(self, graph) -> dict[str, object]:
        out: dict[str, object] = {}
        self._placeholder_cursor = 0
        for node in list(graph.nodes):
            if node.op == "placeholder":
                out[node.name] = self._infer_placeholder_type(node)
                self._placeholder_cursor += 1
            elif node.op == "call_function":
                out[node.name] = self._infer_call_type(node, out)
            elif node.op == "call_module" or node.op == "output":
                continue
            elif node.op == "get_attr":
                raise NotImplementedError("FX get_attr nodes are not yet supported by ForgeTorch.")
            else:
                raise NotImplementedError(
                    f"unsupported FX node.op: {node.op!r} (target={node.target!r})"
                )
        return out

    def _infer_placeholder_type(self, node: Node):
        meta_type = self._tensor_type(node)
        if meta_type is not None:
            return meta_type
        idx = _placeholder_index(node)
        ex = self._example_inputs[idx] if idx < len(self._example_inputs) else None
        torch_dtype = ex.dtype if ex is not None else torch.float32
        shape = tuple(ex.shape) if ex is not None else (0,)
        dtype_str = DTYPE_TO_FORGE.get(torch_dtype)
        if dtype_str is None:
            raise NotImplementedError(
                f"placeholder dtype {torch_dtype!r} is not yet supported by "
                f"the ForgeStack dialects."
            )
        return self._forge_tensor_cls()(list(shape), dtype_str)

    def _infer_call_type(self, node: Node, node_types: dict[str, object]):
        target_name = map_node_to_forge(node.target)
        operand_types = [node_types[n] for n in _data_arg_names(node)]

        meta_type = self._tensor_type(node)
        if meta_type is not None:
            return meta_type

        forge_tensor = self._forge_tensor_cls()
        if target_name == "forge.graph.matmul":
            return _infer_matmul_type(operand_types, forge_tensor)
        if target_name == "forge.graph.add":
            return _infer_add_type(operand_types, forge_tensor)
        if target_name == "forge.graph.gelu":
            operand = _expect_tensor(operand_types[0], forge_tensor)
            return forge_tensor(list(_shape_seq(operand)), _dtype_str(operand))
        if target_name == "forge.graph.view":
            target_shape = _extract_view_target_shape(node)
            if target_shape is None:
                raise NotImplementedError(
                    "aten.view without a concrete shape cannot be lowered "
                    "in Phase 1 (would need symbolic-shape inference)."
                )
            operand = _expect_tensor(operand_types[0], forge_tensor)
            return forge_tensor(list(target_shape), _dtype_str(operand))
        raise AssertionError(f"unknown forge.graph op: {target_name!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Op emission (pass 2)
    # ------------------------------------------------------------------

    def _emit_call_function(
        self,
        builder: Builder,
        node: Node,
        ssa_values: dict[str, SSAValue],
    ) -> None:
        target_name = map_node_to_forge(node.target)
        operands = [ssa_values[n] for n in _data_arg_names(node)]
        result_type = self._result_type_for(node, operands)
        op = _build_op(target_name, operands, result_type)
        builder.insert(op)
        ssa_values[node.name] = op.results[0]

    def _result_type_for(self, node: Node, operands: list):
        meta_type = self._tensor_type(node)
        if meta_type is not None:
            return meta_type
        forge_tensor = self._forge_tensor_cls()
        target_name = map_node_to_forge(node.target)
        if target_name == "forge.graph.matmul":
            return _infer_matmul_type(_operand_types(operands, forge_tensor), forge_tensor)
        if target_name == "forge.graph.add":
            return _infer_add_type(_operand_types(operands, forge_tensor), forge_tensor)
        if target_name == "forge.graph.gelu":
            operand_type = _expect_tensor(operands[0].type, forge_tensor)
            return forge_tensor(list(_shape_seq(operand_type)), _dtype_str(operand_type))
        if target_name == "forge.graph.view":
            target_shape = _extract_view_target_shape(node)
            if target_shape is None:
                raise NotImplementedError(
                    "aten.view without a concrete shape cannot be lowered "
                    "in Phase 1 (would need symbolic-shape inference)."
                )
            operand_type = _expect_tensor(operands[0].type, forge_tensor)
            return forge_tensor(list(target_shape), _dtype_str(operand_type))
        raise AssertionError(f"unknown forge.graph op: {target_name!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tensor_type(self, node: Node):
        meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
        if meta is None:
            return None
        torch_dtype = meta.dtype
        dtype_str = DTYPE_TO_FORGE.get(torch_dtype)
        if dtype_str is None:
            raise NotImplementedError(
                f"FX node {node.name!r}: dtype {torch_dtype!r} is not yet "
                f"supported by the ForgeStack dialects."
            )
        return self._forge_tensor_cls()(list(meta.shape), dtype_str)

    @staticmethod
    def _load_graph_dialect():
        return importlib.import_module("compiler.lib.Dialect.Graph")

    @staticmethod
    def _forge_tensor_cls():
        return FxImporter._load_graph_dialect().TensorType

    @staticmethod
    def _build_context() -> Context:
        ctx = Context(allow_unregistered=False)
        ctx.load_dialect(Builtin)
        from xdsl.dialects.func import Func as FuncDialect

        ctx.load_dialect(FuncDialect)
        ctx.load_dialect(FxImporter._load_graph_dialect().Graph)
        tensor_mod = importlib.import_module("compiler.lib.Dialect.Tensor")
        ctx.load_dialect(tensor_mod.Tensor)
        return ctx


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _placeholder_index(node: Node) -> int:
    count = 0
    for n in node.graph.nodes:  # type: ignore[attr-defined]
        if n is node:
            return count
        if n.op == "placeholder":
            count += 1
    return count  # pragma: no cover — node is always reachable


def _expect_tensor(value, forge_tensor):
    if not isinstance(value, forge_tensor):  # type: ignore[arg-type]
        raise TypeError(f"operand has unexpected type {value!r}; expected {forge_tensor.__name__}")
    return value


def _ssa_type(ssa_value: SSAValue, forge_tensor):
    return ssa_value.type


def _operand_types(operands: list, forge_tensor) -> list:
    return [_expect_tensor(operand.type, forge_tensor) for operand in operands]  # type: ignore[union-attr]


def _shape_seq(tensor_type) -> tuple[int, ...]:
    return tuple(int(d.data) for d in tensor_type.shape.data)


def _dtype_str(tensor_type) -> str:
    return tensor_type.dtype.data.value


def _broadcast(shape_a: list, shape_b: list):
    ndim = max(len(shape_a), len(shape_b))
    pad_a = [1] * (ndim - len(shape_a)) + list(shape_a)
    pad_b = [1] * (ndim - len(shape_b)) + list(shape_b)
    out: list = []
    for a, b in zip(pad_a, pad_b, strict=True):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            return None
    return out


def _infer_matmul_type(operand_types: list, forge_tensor):
    a = _expect_tensor(operand_types[0], forge_tensor)
    b = _expect_tensor(operand_types[1], forge_tensor)
    shape_a = list(_shape_seq(a))
    shape_b = list(_shape_seq(b))
    if len(shape_a) < 2 or len(shape_b) < 2:
        raise ValueError(f"forge.graph.matmul needs rank>=2 operands, got {shape_a} and {shape_b}")
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"matmul inner-dim mismatch: {shape_a[-1]} vs {shape_b[-2]}")
    batch = _broadcast(shape_a[:-2], shape_b[:-2])
    if batch is None:
        raise ValueError(
            f"matmul batch dims are not broadcastable: {shape_a[:-2]} vs {shape_b[:-2]}"
        )
    out_shape = (*batch, shape_a[-2], shape_b[-1])
    return forge_tensor(list(out_shape), _dtype_str(a))


def _infer_add_type(operand_types: list, forge_tensor):
    a = _expect_tensor(operand_types[0], forge_tensor)
    b = _expect_tensor(operand_types[1], forge_tensor)
    shape_a = list(_shape_seq(a))
    shape_b = list(_shape_seq(b))
    out_shape = _broadcast(shape_a, shape_b)
    if out_shape is None:
        raise ValueError(f"add: shapes {shape_a} and {shape_b} are not broadcastable")
    return forge_tensor(list(out_shape), _dtype_str(a))


def _arg_name(arg) -> str:
    """FX graph ``Node.args`` entries are :class:`Node` or Python literals.

    The :class:`output` node is the only place this helper is still useful —
    call_function args always go through :func:`_data_arg_names`, which
    filters out non-Node entries (constant lists like ``aten.view``'s size).
    """
    if isinstance(arg, Node):
        return arg.name
    raise TypeError(
        f"unexpected FX argument: {arg!r}; only Node placeholders are "
        "supported by the shipped FX importer (constants and Python ints "
        "are not yet lowered)."
    )


def _data_arg_names(node: Node) -> list[str]:
    """Names of the data (tensor) SSAValues flowing into ``node``.

    ``torch.fx`` stores a call_function's args as ``tuple[Argument, ...]``
    where each entry is either a :class:`Node` (data dependency) or a Python
    literal (constant / kwarg-passed shape list). Only the Nodes correspond
    to ``forge.tensor`` SSAValues; everything else is filtered here so the
    downstream ``ssa_values`` lookup is unambiguous.
    """
    names = []
    for arg in node.args:
        if isinstance(arg, Node):
            names.append(arg.name)
        elif isinstance(arg, (tuple, list)):
            continue
    return names


def _flatten_output_args(args):
    """FX ``output`` nodes may carry a single Node or a tuple/list of them
    (the latter is what ``torch.export`` emits to model multi-result outputs)."""
    flat = []
    for arg in args:
        if isinstance(arg, (tuple, list)):
            flat.extend(arg)
        else:
            flat.append(arg)
    return flat


def _output_node(graph):
    for node in graph.nodes:  # type: ignore[attr-defined]
        if node.op == "output":
            return node
    return None


def _extract_view_target_shape(node: Node) -> list[int] | None:
    args = list(node.args)
    if len(args) < 2:
        return None
    target = args[1]
    if isinstance(target, (list, tuple)) and all(isinstance(d, int) for d in target):
        return [int(d) for d in target]
    return None


def _build_op(target_name: str, operands: list, result_type):
    graph_mod = FxImporter._load_graph_dialect()
    if target_name == "forge.graph.matmul":
        return graph_mod.MatmulOp.create(operands=operands, result_types=[result_type])
    if target_name == "forge.graph.add":
        return graph_mod.AddOp.create(operands=operands, result_types=[result_type])
    if target_name == "forge.graph.gelu":
        return graph_mod.GeluOp.create(operands=operands, result_types=[result_type])
    if target_name == "forge.graph.view":
        return graph_mod.ViewOp.create(operands=operands, result_types=[result_type])
    raise AssertionError(f"_build_op: unknown forge.graph op {target_name!r}")  # pragma: no cover
