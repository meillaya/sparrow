"""FX → MLIR importer for Sparrow.

The :class:`FxImporter` walks a captured ``torch.fx.GraphModule`` and emits
``forge.graph`` / ``forge.tensor`` dialect MLIR. It is the bridge between
PyTorch's symbolic graph capture and the xDSL dialect surface defined
under :mod:`compiler.lib.Dialect`.

Two capture styles are supported:

* ``torch.fx.symbolic_trace`` — parameter tensors appear as ``get_attr``
  nodes and shape arithmetic appears as ``getattr`` / ``getitem`` nodes.
  The importer lifts parameters into extra function arguments and folds
  shape expressions using the example inputs.
* ``make_fx`` / Dynamo — parameters are already graph placeholders and
  ``tensor_meta`` carries concrete shapes.

The op translation table lives in :mod:`forge_torch.ops`; this module
only orchestrates type inference and the xDSL IRBuilder.
"""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

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
from xdsl.dialects.builtin import BoolAttr, Builtin, ModuleOp  # noqa: E402
from xdsl.dialects.func import FuncOp, ReturnOp  # noqa: E402
from xdsl.ir import Block, Region, SSAValue  # noqa: E402
from xdsl.printer import Printer  # noqa: E402

from forge_torch.ops import (  # noqa: E402
    DTYPE_TO_FORGE,
    GELU,
    GEMM,
    RESHAPE,
    SELECT,
    TRANSPOSE,
    OpSpec,
    normalize_target,
    spec_for_aten,
    spec_for_method,
)


def _default_function_name(gm: GraphModule) -> str:
    name = getattr(gm, "_graph_name", None) or gm.__class__.__name__
    clean = str(name).replace(".", "_").replace(" ", "_")
    return f"{clean}_graph"


class FxImporter:
    """Walk a ``torch.fx.GraphModule`` and emit forge dialect MLIR."""

    def __init__(
        self,
        example_inputs: list[torch.Tensor] | None = None,
        *,
        symbolic_heuristic: bool = False,
    ) -> None:
        self._example_inputs: list[torch.Tensor] = list(example_inputs or [])
        # When True, 3D inputs (FusedMLP-style) with no explicit symbolic-dim
        # metadata get ``B, S, <concrete>`` as their shape — matches the
        # spec §3 ``B >= 1, S >= 1`` example. Opt-in so the existing
        # concrete-shape tests (Task 8, Phase 1) keep passing.
        self._symbolic_heuristic = symbolic_heuristic
        #: Parameter / buffer tensors lifted into function arguments, in
        #: argument order after the user placeholders.
        self.lifted_tensors: list[torch.Tensor] = []
        self._lifted: dict[str, torch.Tensor] = {}
        self._lifted_order: list[str] = []
        self._lifted_ssa: dict[str, SSAValue] = {}
        self._node_lifted_key: dict[str, str] = {}
        self._shape_values: dict[str, object] = {}
        self._node_types: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def import_graph(self, gm: GraphModule) -> str:
        """Import ``gm`` and return textual MLIR."""
        module = self.import_module(gm)
        buf = io.StringIO()
        Printer(stream=buf).print_op(module)
        return buf.getvalue()

    def import_module(self, gm: GraphModule) -> ModuleOp:
        """Import ``gm`` and return the verified xDSL module."""
        graph = gm.graph if hasattr(gm, "graph") else gm
        self._lifted = {}
        self._lifted_order = []
        self._lifted_ssa = {}
        self._node_lifted_key = {}
        self._shape_values = {}
        node_types = self._infer_node_types(graph, gm)
        self._node_types = node_types

        placeholder_names: list[str] = []
        placeholder_types: list = []
        for node in list(graph.nodes):
            if node.op == "placeholder":
                placeholder_names.append(node.name)
                placeholder_types.append(node_types[node.name])

        lifted_names = list(self._lifted_order)
        lifted_types = [self._tensor_type_for_tensor(self._lifted[name]) for name in lifted_names]

        output_node = _output_node(graph)
        if output_node is None:
            output_arg_names: list[str] = []
            output_types: list = []
        else:
            flat_args = _flatten_output_args(output_node.args)  # type: ignore[arg-type]
            output_arg_names = [_arg_name(a) for a in flat_args]
            output_types = [node_types[name] for name in output_arg_names]

        all_input_types = placeholder_types + lifted_types
        block = Block(arg_types=all_input_types)
        func = FuncOp.from_region(
            _default_function_name(gm),
            input_types=all_input_types,
            return_types=output_types,
            region=Region(block),
        )

        ssa_values: dict[str, SSAValue] = {}
        for arg, name in zip(block.args[: len(placeholder_names)], placeholder_names, strict=True):
            ssa_values[name] = arg
        for arg, name in zip(block.args[len(placeholder_names) :], lifted_names, strict=True):
            self._lifted_ssa[name] = arg

        builder = Builder(InsertPoint.at_start(block))
        for node in list(graph.nodes):
            if node.op in ("placeholder", "output", "get_attr"):
                continue
            if node.name in self._shape_values:
                continue
            if node.op == "call_function":
                self._emit_call_function(builder, node, ssa_values)
                continue
            if node.op == "call_method":
                self._emit_call_method(builder, node, ssa_values)
                continue
            if node.op == "call_module":
                self._emit_call_module(builder, node, ssa_values, gm)
                continue
            raise AssertionError(f"unhandled FX node.op: {node.op!r}")  # pragma: no cover

        if output_node is not None:
            operands = [self._lookup_ssa(name, ssa_values) for name in output_arg_names]
            builder.insert(ReturnOp.create(operands=operands, result_types=[]))
        elif output_types:
            builder.insert(ReturnOp.create(operands=[], result_types=[]))

        func.verify()
        module = ModuleOp([func])
        module.verify()
        self.lifted_tensors = [self._lifted[name] for name in lifted_names]
        return module

    # ------------------------------------------------------------------
    # Type inference (pass 1)
    # ------------------------------------------------------------------

    def _infer_node_types(self, graph, gm: GraphModule) -> dict[str, object]:
        out: dict[str, object] = {}
        self._node_types = out
        self._placeholder_cursor = 0
        for node in list(graph.nodes):
            if node.op == "placeholder":
                out[node.name] = self._infer_placeholder_type(node)
                self._placeholder_cursor += 1
            elif node.op == "get_attr":
                out[node.name] = self._infer_get_attr_type(node, gm)
            elif node.op == "call_function":
                if self._try_shape_expr(node):
                    continue
                out[node.name] = self._infer_call_type(node, out, gm)
            elif node.op == "call_method":
                if self._try_shape_expr(node):
                    continue
                out[node.name] = self._infer_method_type(node, out)
            elif node.op == "call_module":
                out[node.name] = self._infer_call_module_type(node, out, gm)
            elif node.op == "output":
                continue
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
                f"placeholder dtype {torch_dtype!r} is not yet supported by the Sparrow dialects."
            )
        return self._forge_tensor_cls()(list(shape), dtype_str)

    # -- shape expressions -------------------------------------------------

    def _try_shape_expr(self, node: Node) -> bool:
        """Record ``node`` as static shape metadata and return ``True``."""
        if node.op == "call_method":
            name = str(node.target)
            if name == "size":
                base = _arg_node(node.args[0]) if node.args else None
                if base is None or base.name not in self._node_types:
                    return False
                shape = self._shape_of_type(self._node_types[base.name])
                if len(node.args) == 1:
                    self._shape_values[node.name] = tuple(shape)
                else:
                    dim = _as_int(node.args[1])
                    if dim is None:
                        return False
                    self._shape_values[node.name] = shape[_normalize_dim(dim, len(shape))]
                return True
            if name == "dim":
                base = _arg_node(node.args[0]) if node.args else None
                if base is None or base.name not in self._node_types:
                    return False
                self._shape_values[node.name] = len(
                    self._shape_of_type(self._node_types[base.name])
                )
                return True
            if name == "numel":
                base = _arg_node(node.args[0]) if node.args else None
                if base is None or base.name not in self._node_types:
                    return False
                self._shape_values[node.name] = _numel(
                    self._shape_of_type(self._node_types[base.name])
                )
                return True
            return False

        if node.op != "call_function":
            return False
        target = node.target
        normalized = normalize_target(target)
        if normalized in {"aten.sym_size", "aten.sym_numel"}:
            base = _arg_node(node.args[0]) if node.args else None
            if base is None or base.name not in self._node_types:
                return False
            shape = self._shape_of_type(self._node_types[base.name])
            if normalized == "aten.sym_numel":
                self._shape_values[node.name] = _numel(shape)
            elif len(node.args) > 1:
                dim = self._resolve_int(node.args[1])
                self._shape_values[node.name] = shape[_normalize_dim(dim, len(shape))]
            return True
        raw = str(target)
        if "getattr" in raw:
            if len(node.args) < 2:
                return False
            base = _arg_node(node.args[0])
            attr = node.args[1]
            if base is None or attr != "shape" or base.name not in self._node_types:
                return False
            self._shape_values[node.name] = tuple(self._shape_of_type(self._node_types[base.name]))
            return True
        if "getitem" in raw:
            if len(node.args) < 2:
                return False
            base = _arg_node(node.args[0])
            if base is None or base.name not in self._shape_values:
                return False
            value = self._shape_values[base.name]
            index = self._resolve_int(node.args[1])
            if not isinstance(value, tuple):
                return False
            self._shape_values[node.name] = value[index]
            return True
        if normalized in _INT_OPS:
            values: list[int] = []
            for arg in node.args:
                value = self._resolve_int_maybe(arg)
                if value is None:
                    return False
                values.append(value)
            self._shape_values[node.name] = _apply_int_op(normalized, values)
            return True
        return False

    def _resolve_int(self, value: object) -> int:
        resolved = self._resolve_int_maybe(value)
        if resolved is None:
            raise NotImplementedError(
                f"cannot resolve {value!r} to a static integer; Sparrow needs "
                "concrete shapes for the CPU plan"
            )
        return resolved

    def _resolve_int_maybe(self, value: object) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, Node):
            stored = self._shape_values.get(value.name)
            if isinstance(stored, int):
                return stored
            target = str(value.target)
            if value.op == "call_function" and "getitem" in target and len(value.args) >= 2:
                base = _arg_node(value.args[0])
                if base is not None and base.name in self._shape_values:
                    container = self._shape_values[base.name]
                    index = self._resolve_int_maybe(value.args[1])
                    if index is not None and isinstance(container, tuple):
                        item = container[index]
                        return item if isinstance(item, int) else None
        return None

    def _resolve_shape(self, value: object) -> list[int]:
        """Resolve a reshape/permute target to concrete ints."""
        if isinstance(value, Node):
            stored = self._shape_values.get(value.name)
            if stored is not None:
                value = stored
        if isinstance(value, (tuple, list)):
            dims: list[int] = []
            for item in value:
                dims.append(self._resolve_int(item))
            return dims
        return [self._resolve_int(value)]

    # -- tensor ops --------------------------------------------------------

    def _infer_call_type(self, node: Node, node_types: dict[str, object], gm: GraphModule):
        self._check_supported_scalars(node)
        meta_type = self._tensor_type(node)
        if meta_type is not None:
            return meta_type
        spec = spec_for_aten(node.target)
        operands = [node_types[name] for name in self._operand_names(node)]
        normalized = normalize_target(node.target)
        return self._infer_spec_type(spec, node, operands, normalized)

    @staticmethod
    def _check_supported_scalars(node: Node) -> None:
        """Reject aten forms whose scalar modifiers the lowering ignores."""
        if normalize_target(node.target) != "aten.addmm":
            return
        kwargs = getattr(node, "kwargs", {}) or {}
        for name in ("alpha", "beta"):
            value = kwargs.get(name, 1)
            if value != 1:
                raise NotImplementedError(
                    f"aten.addmm {name}={value!r} is not supported; the CPU "
                    "lowering implements the default alpha=1, beta=1 form"
                )

    def _infer_method_type(self, node: Node, node_types: dict[str, object]):
        meta_type = self._tensor_type(node)
        if meta_type is not None:
            return meta_type
        try:
            spec = spec_for_method(str(node.target))
        except KeyError as exc:
            raise NotImplementedError(
                f"call_method {node.target!r} is not lowered by the Sparrow importer"
            ) from exc
        operands = [node_types[name] for name in self._operand_names(node)]
        return self._infer_spec_type(spec, node, operands, f"method.{node.target}")

    def _infer_spec_type(
        self,
        spec: OpSpec,
        node: Node,
        operand_types: list,
        variant: str,
    ):
        forge_tensor = self._forge_tensor_cls()
        if spec.op == GEMM.op:
            normalized = variant if variant.startswith("aten.") else normalize_target(node.target)
            if normalized == "aten.linear":
                return _infer_linear_type(operand_types, forge_tensor)
            if normalized == "aten.addmm" or variant == "method.addmm":
                if len(operand_types) < 3:
                    raise ValueError("aten.addmm needs bias, a, and b operands")
                return _infer_matmul_type(operand_types[1:], forge_tensor)
            return _infer_matmul_type(operand_types, forge_tensor)
        if spec.op in {"forge.tensor.broadcast_add", "forge.tensor.mul"}:
            return _infer_broadcast_binary(operand_types, forge_tensor)
        if spec.op in {"forge.tensor.gelu", "forge.tensor.relu", "forge.tensor.layer_norm"}:
            operand = _expect_tensor(operand_types[0], forge_tensor)
            return forge_tensor(list(_shape_seq(operand)), _dtype_str(operand))
        if spec.op == "forge.tensor.softmax":
            operand = _expect_tensor(operand_types[0], forge_tensor)
            return forge_tensor(list(_shape_seq(operand)), _dtype_str(operand))
        if spec.op == "forge.tensor.scaled_dot_product_attention":
            operand = _expect_tensor(operand_types[0], forge_tensor)
            return forge_tensor(list(_shape_seq(operand)), _dtype_str(operand))
        if spec.op == RESHAPE.op:
            operand = _expect_tensor(operand_types[0], forge_tensor)
            operand_shape = list(_shape_seq(operand))
            if any(isinstance(d, str) for d in operand_shape):
                return forge_tensor(operand_shape, _dtype_str(operand))
            target_shape = _extract_reshape_target(node, self._resolve_shape, allow_symbolic=True)
            if target_shape is None:
                raise NotImplementedError(
                    "reshape without a concrete target shape cannot be lowered to the CPU plan"
                )
            return forge_tensor(list(target_shape), _dtype_str(operand))
        if spec.op == TRANSPOSE.op:
            operand = _expect_tensor(operand_types[0], forge_tensor)
            shape = list(_shape_seq(operand))
            permutation = self._permutation_for(node, len(shape))
            return forge_tensor([shape[i] for i in permutation], _dtype_str(operand))
        if spec.op == SELECT.op:
            operand = _expect_tensor(operand_types[0], forge_tensor)
            shape = list(_shape_seq(operand))
            dim = self._select_dim_for(node, len(shape))
            return forge_tensor(shape[:dim] + shape[dim + 1 :], _dtype_str(operand))
        raise AssertionError(f"no shape rule for {spec.op}")  # pragma: no cover

    def _infer_call_module_type(self, node: Node, node_types: dict[str, object], gm: GraphModule):
        meta_type = self._tensor_type(node)
        if meta_type is not None:
            return meta_type
        module = gm.get_submodule(str(node.target))
        operand_names = self._operand_names(node)
        if not operand_names:
            raise NotImplementedError(f"call_module {node.target!r} has no tensor input")
        operand = _expect_tensor(node_types[operand_names[0]], self._forge_tensor_cls())
        shape = list(_shape_seq(operand))
        dtype = _dtype_str(operand)
        if isinstance(module, torch.nn.Linear):
            self._register_lifted(f"{node.target}.weight", module.weight)
            if module.bias is not None:
                self._register_lifted(f"{node.target}.bias", module.bias)
            return self._forge_tensor_cls()([*shape[:-1], module.out_features], dtype)
        if isinstance(module, torch.nn.LayerNorm):
            if module.weight is None or module.bias is None:
                raise NotImplementedError(
                    f"call_module target {node.target!r} is an nn.LayerNorm without "
                    "affine parameters; the CPU layer-norm op requires gamma and beta"
                )
            self._register_lifted(f"{node.target}.weight", module.weight)
            self._register_lifted(f"{node.target}.bias", module.bias)
            return self._forge_tensor_cls()(shape, dtype)
        raise NotImplementedError(
            f"call_module target {node.target!r} ({type(module).__name__}) is not "
            "supported; only nn.Linear and nn.LayerNorm are lowered"
        )

    def _infer_get_attr_type(self, node: Node, gm: GraphModule):
        value = _lookup_attr(gm, str(node.target))
        if not isinstance(value, torch.Tensor):
            raise NotImplementedError(
                f"get_attr target {node.target!r} is not a tensor; only parameter "
                "and buffer tensors can be lifted into the plan"
            )
        key = str(node.target)
        self._register_lifted(key, value)
        self._node_lifted_key[node.name] = key
        return self._tensor_type_for_tensor(value)

    # ------------------------------------------------------------------
    # Op emission (pass 2)
    # ------------------------------------------------------------------

    def _emit_call_function(
        self,
        builder: Builder,
        node: Node,
        ssa_values: dict[str, SSAValue],
    ) -> None:
        spec = spec_for_aten(node.target)
        operands = self._ssa_operands(node, ssa_values)
        result_type = self._node_types[node.name]
        normalized = normalize_target(node.target)

        if spec.op == GEMM.op:
            value = self._emit_gemm(builder, node, operands, result_type, normalized)
            ssa_values[node.name] = value
            return
        if spec.op == RESHAPE.op:
            graph_mod = self._load_graph_dialect()
            op = graph_mod.ViewOp.create(operands=operands[:1], result_types=[result_type])
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        if spec.op == TRANSPOSE.op:
            tensor_mod = self._load_tensor_dialect()
            permutation = self._permutation_for(node, len(_shape_seq(result_type)))
            op = tensor_mod.TransposeOp.create(
                operands=operands[:1],
                result_types=[result_type],
                properties={
                    "permutation": tensor_mod.ArrayAttr(
                        [tensor_mod.IntAttr(int(d)) for d in permutation]
                    )
                },
            )
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        if spec.op == SELECT.op:
            tensor_mod = self._load_tensor_dialect()
            dim, index = self._select_index_for(node)
            op = tensor_mod.SelectOp.create(
                operands=operands[:1],
                result_types=[result_type],
                properties={
                    "dim": tensor_mod.IntAttr(int(dim)),
                    "index": tensor_mod.IntAttr(int(index)),
                },
            )
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        op = _build_op(spec, operands, result_type, node)
        builder.insert(op)
        ssa_values[node.name] = op.results[0]

    def _emit_call_method(
        self,
        builder: Builder,
        node: Node,
        ssa_values: dict[str, SSAValue],
    ) -> None:
        spec = spec_for_method(str(node.target))
        operands = self._ssa_operands(node, ssa_values)
        result_type = self._node_types[node.name]
        if spec.op == GEMM.op:
            value = self._emit_gemm(builder, node, operands, result_type, f"method.{node.target}")
            ssa_values[node.name] = value
            return
        if spec.op == RESHAPE.op:
            graph_mod = self._load_graph_dialect()
            op = graph_mod.ViewOp.create(operands=operands[:1], result_types=[result_type])
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        if spec.op == TRANSPOSE.op:
            tensor_mod = self._load_tensor_dialect()
            shape = _shape_seq(result_type)
            permutation = self._permutation_for(node, len(shape))
            op = tensor_mod.TransposeOp.create(
                operands=operands[:1],
                result_types=[result_type],
                properties={
                    "permutation": tensor_mod.ArrayAttr(
                        [tensor_mod.IntAttr(int(d)) for d in permutation]
                    )
                },
            )
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        if spec.op == "forge.tensor.broadcast_add":
            tensor_mod = self._load_tensor_dialect()
            op = tensor_mod.BroadcastAddOp.create(operands=operands, result_types=[result_type])
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        if spec.op == "forge.tensor.mul":
            tensor_mod = self._load_tensor_dialect()
            op = tensor_mod.MulOp.create(operands=operands, result_types=[result_type])
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        raise AssertionError(f"no method lowering for {spec.op}")  # pragma: no cover

    def _emit_call_module(
        self,
        builder: Builder,
        node: Node,
        ssa_values: dict[str, SSAValue],
        gm: GraphModule,
    ) -> None:
        module = gm.get_submodule(str(node.target))
        operands = self._ssa_operands(node, ssa_values)
        result_type = self._node_types[node.name]
        tensor_mod = self._load_tensor_dialect()
        if isinstance(module, torch.nn.Linear):
            weight = self._lifted_ssa[f"{node.target}.weight"]
            gemm = tensor_mod.GemmOp.create(
                operands=[operands[0], weight],
                result_types=[result_type],
                properties={"transpose_b": BoolAttr.from_bool(True)},
            )
            builder.insert(gemm)
            value = gemm.results[0]
            if module.bias is not None:
                bias = self._lifted_ssa[f"{node.target}.bias"]
                add = tensor_mod.BroadcastAddOp.create(
                    operands=[value, bias], result_types=[result_type]
                )
                builder.insert(add)
                value = add.results[0]
            ssa_values[node.name] = value
            return
        if isinstance(module, torch.nn.LayerNorm):
            weight = self._lifted_ssa[f"{node.target}.weight"]
            bias = self._lifted_ssa[f"{node.target}.bias"]
            op = tensor_mod.LayerNormOp.create(
                operands=[operands[0], weight, bias],
                result_types=[result_type],
                properties={
                    "epsilon": tensor_mod.FloatAttr(float(module.eps), tensor_mod.Float32Type()),
                },
            )
            builder.insert(op)
            ssa_values[node.name] = op.results[0]
            return
        raise AssertionError(f"no call_module lowering for {type(module).__name__}")

    def _emit_gemm(
        self,
        builder: Builder,
        node: Node,
        operands: list[SSAValue],
        result_type,
        variant: str,
    ) -> SSAValue:
        """Emit a gemm plus an optional bias add for linear/addmm."""
        graph_mod = self._load_graph_dialect()
        tensor_mod = self._load_tensor_dialect()
        normalized = variant if variant.startswith("aten.") else normalize_target(node.target)
        if normalized == "aten.linear":
            a, b = operands[0], operands[1]
            bias = operands[2] if len(operands) > 2 else None
            gemm = tensor_mod.GemmOp.create(
                operands=[a, b],
                result_types=[result_type],
                properties={"transpose_b": BoolAttr.from_bool(True)},
            )
            builder.insert(gemm)
            value = gemm.results[0]
            if bias is not None:
                add = tensor_mod.BroadcastAddOp.create(
                    operands=[value, bias], result_types=[result_type]
                )
                builder.insert(add)
                value = add.results[0]
            return value
        if normalized == "aten.addmm" or variant == "method.addmm":
            if len(operands) < 3:
                raise NotImplementedError("aten.addmm needs a bias operand")
            bias, a, b = operands[0], operands[1], operands[2]
            gemm = tensor_mod.GemmOp.create(operands=[a, b], result_types=[result_type])
            builder.insert(gemm)
            add = tensor_mod.BroadcastAddOp.create(
                operands=[gemm.results[0], bias], result_types=[result_type]
            )
            builder.insert(add)
            return add.results[0]
        op = graph_mod.MatmulOp.create(operands=operands[:2], result_types=[result_type])
        builder.insert(op)
        return op.results[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _operand_names(self, node: Node) -> list[str]:
        """Names of the tensor SSA values flowing into ``node``.

        Shape bookkeeping nodes are filtered out (they have no SSA value),
        as are Python literals such as reshape sizes. Kwargs are scanned
        because ``aten.layer_norm`` passes weight and bias there.
        """
        names: list[str] = []
        for arg in node.args:
            if isinstance(arg, Node) and arg.name not in self._shape_values:
                names.append(arg.name)
            elif isinstance(arg, (tuple, list)):
                for item in arg:
                    if isinstance(item, Node) and item.name not in self._shape_values:
                        names.append(item.name)
        for arg in getattr(node, "kwargs", {}).values():
            if isinstance(arg, (tuple, list)):
                for item in arg:
                    if isinstance(item, Node) and item.name not in self._shape_values:
                        names.append(item.name)
            elif isinstance(arg, Node) and arg.name not in self._shape_values:
                names.append(arg.name)
        return names

    def _ssa_operands(self, node: Node, ssa_values: dict[str, SSAValue]) -> list[SSAValue]:
        return [self._lookup_ssa(name, ssa_values) for name in self._operand_names(node)]

    def _lookup_ssa(self, name: str, ssa_values: dict[str, SSAValue]) -> SSAValue:
        value = ssa_values.get(name)
        if value is not None:
            return value
        key = self._node_lifted_key.get(name)
        if key is None:
            raise KeyError(f"no SSA value for {name!r}")
        return self._lifted_ssa[key]

    def _permutation_for(self, node: Node, rank: int) -> list[int]:
        target = str(node.target)
        normalized = normalize_target(node.target) if node.op == "call_function" else target
        if normalized == "aten.t" or target == "t":
            return list(reversed(range(rank)))
        if normalized in {"aten.permute"} or target == "permute":
            dims: object = node.args[1] if len(node.args) == 2 else list(node.args[1:])
            return [_normalize_dim(self._resolve_int(d), rank) for d in _as_dims(dims)]
        dims_pair = list(node.args[1:3])
        if len(dims_pair) == 2:
            dim0 = _normalize_dim(self._resolve_int(dims_pair[0]), rank)
            dim1 = _normalize_dim(self._resolve_int(dims_pair[1]), rank)
            permutation = list(range(rank))
            permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
            return permutation
        raise NotImplementedError(f"cannot derive a permutation for {node!r}")

    def _select_index_for(self, node: Node) -> tuple[int, int]:
        normalized = (
            normalize_target(node.target) if node.op == "call_function" else str(node.target)
        )
        if normalized == "aten.select":
            shape = self._shape_of_type(self._node_types[self._operand_names(node)[0]])
            dim = _normalize_dim(self._resolve_int(node.args[1]), len(shape))
            index = self._resolve_int(node.args[2])
            return dim, index
        # ``operator.getitem`` / ``aten.getitem``: one integer index on dim 0.
        index = self._resolve_int(node.args[1])
        return 0, index

    def _select_dim_for(self, node: Node, rank: int) -> int:
        normalized = (
            normalize_target(node.target) if node.op == "call_function" else str(node.target)
        )
        if normalized == "aten.select":
            return _normalize_dim(self._resolve_int(node.args[1]), rank)
        return 0

    def _tensor_type_for_tensor(self, tensor: torch.Tensor):
        dtype_str = DTYPE_TO_FORGE.get(tensor.dtype)
        if dtype_str is None:
            raise NotImplementedError(
                f"parameter dtype {tensor.dtype!r} is not supported by the Sparrow dialects."
            )
        return self._forge_tensor_cls()(list(tensor.shape), dtype_str)

    def _register_lifted(self, key: str, tensor: torch.Tensor) -> None:
        if key not in self._lifted:
            self._lifted[key] = tensor
            self._lifted_order.append(key)

    def _shape_of_type(self, type_attr: object) -> tuple:
        return _shape_seq(type_attr)

    def _tensor_type(self, node: Node):
        meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
        if meta is None:
            return None
        # ``tensor_meta`` may be a torch.fx TensorMetadata dataclass (shape,
        # dtype attrs) or a plain dict (test-friendly). Cover both.
        if isinstance(meta, dict):
            torch_dtype = meta["dtype"]
            shape = list(meta["shape"])
        else:
            torch_dtype = meta.dtype
            shape = list(meta.shape)
        dtype_str = DTYPE_TO_FORGE.get(torch_dtype)
        if dtype_str is None:
            raise NotImplementedError(
                f"FX node {node.name!r}: dtype {torch_dtype!r} is not yet "
                f"supported by the Sparrow dialects."
            )
        symbolic = self._symbolic_dim_for(node, shape)
        if symbolic is not None:
            sym_dims, constraints = symbolic
            return self._forge_tensor_cls()(sym_dims, dtype_str, constraints=constraints)
        return self._forge_tensor_cls()(shape, dtype_str)

    def _symbolic_dim_for(
        self, node: Node, shape: list[int]
    ) -> tuple[list[object], list[object]] | None:
        """Return ``(sym_dims, constraints)`` if ``node`` carries symbolic-dim metadata.

        Reads symbolic-dim names from one of (priority order):

        1. ``node.meta["tensor_meta"]["symbolic_dim"]`` — the per-spec §3 key.
        2. ``node.meta["symbolic_dim"]`` — the canonical ``torch.export`` /
           dynamo top-level key (fallback for compatibility).

        Value shape: a string (applies to every dim) or a list of
        per-dim names where ``None`` entries render as anonymous ``?``
        and string entries render as named symbolic dims. Falls back to
        the FusedMLP-style heuristic (``B, S, <concrete>`` for 3D inputs)
        when :attr:`_symbolic_heuristic` is set. Returns ``None`` to fall
        through to concrete dim emission.
        """
        meta = getattr(node, "meta", None) or {}
        tensor_meta = meta.get("tensor_meta")
        symbolic_meta: object | None = None
        if tensor_meta is not None:
            if isinstance(tensor_meta, dict):
                symbolic_meta = tensor_meta.get("symbolic_dim")
            else:
                symbolic_meta = getattr(tensor_meta, "symbolic_dim", None)
        if symbolic_meta is None:
            symbolic_meta = meta.get("symbolic_dim")
        sym_names: list[str | None] | None = None
        none_means_concrete = False
        if symbolic_meta is not None:
            if isinstance(symbolic_meta, str):
                sym_names = [symbolic_meta] * len(shape)
            elif isinstance(symbolic_meta, (list, tuple)):
                sym_names = list(symbolic_meta)
            else:
                raise NotImplementedError(
                    f"FX node {node.name!r}: tensor_meta['symbolic_dim'] "
                    f"must be a string or a list of strings, got {type(symbolic_meta).__name__}"
                )
        elif self._symbolic_heuristic and len(shape) == 3:
            sym_names = ["B", "S"] + [None] * (len(shape) - 2)
            none_means_concrete = True

        if sym_names is None:
            return None

        graph_mod = self._load_graph_dialect()
        IntAttr = graph_mod.IntAttr
        StringAttr = graph_mod.StringAttr
        SymbolicConstraint = graph_mod.SymbolicConstraint
        symbolic_mod = importlib.import_module("compiler.lib.Dialect.Graph.symbolic")
        SymbolicConstraintKind = symbolic_mod.SymbolicConstraintKind

        sym_dims: list[object] = []
        constraints: list[object] = []
        for orig_dim, name in zip(shape, sym_names, strict=True):
            if name is None:
                if none_means_concrete:
                    sym_dims.append(IntAttr(int(orig_dim)))
                else:
                    sym_dims.append(StringAttr("?"))
            else:
                sym_dims.append(StringAttr(name))
                constraints.append(
                    SymbolicConstraint(name, SymbolicConstraintKind.GREATER_EQUAL, 1)
                )
        return sym_dims, constraints

    @staticmethod
    def _load_graph_dialect():
        return importlib.import_module("compiler.lib.Dialect.Graph")

    @staticmethod
    def _load_tensor_dialect():
        return importlib.import_module("compiler.lib.Dialect.Tensor")

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

_INT_OPS = frozenset(
    {
        "aten.add",
        "aten.sub",
        "aten.mul",
        "aten.floordiv",
        "aten.truediv",
        "aten.mod",
        "aten.pow",
    }
)


def _apply_int_op(name: str, values: list[int]) -> int:
    if len(values) == 1:
        return values[0]
    left, right = values[0], values[1]
    if name == "aten.add":
        return left + right
    if name == "aten.sub":
        return left - right
    if name == "aten.mul":
        return left * right
    if name == "aten.floordiv":
        return left // right
    if name == "aten.truediv":
        return left // right
    if name == "aten.mod":
        return left % right
    if name == "aten.pow":
        return left**right
    raise AssertionError(name)  # pragma: no cover


def _arg_node(value: object) -> Node | None:
    return value if isinstance(value, Node) else None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _as_dims(value: object) -> list:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _normalize_dim(dim: int, rank: int) -> int:
    resolved = dim + rank if dim < 0 else dim
    if not (0 <= resolved < rank):
        raise ValueError(f"dimension {dim} is out of range for rank {rank}")
    return resolved


def _numel(shape: tuple) -> int:
    total = 1
    for dim in shape:
        if not isinstance(dim, int):
            raise NotImplementedError(f"cannot compute numel of symbolic shape {shape}")
        total *= dim
    return total


def _lookup_attr(gm: GraphModule, target: str) -> object:
    try:
        return gm.get_parameter(target)
    except AttributeError:
        pass
    try:
        return gm.get_buffer(target)
    except AttributeError:
        pass
    value: object = gm
    try:
        for part in target.split("."):
            value = getattr(value, part)
    except AttributeError as exc:
        raise NotImplementedError(
            f"get_attr target {target!r} does not resolve to an attribute"
        ) from exc
    return value


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


def _shape_seq(tensor_type) -> tuple:
    """Return the raw dim values, preserving symbolic dims as strings."""
    out: list = []
    for d in tensor_type.shape.data:
        v = d.data
        if isinstance(v, str):
            out.append(v)
        else:
            out.append(int(v))
    return tuple(out)


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
        elif b == 1 or (isinstance(a, str) and a == b):
            out.append(a)
        elif isinstance(b, str) and a == 1:
            out.append(b)
        elif (isinstance(a, str) and b == 1) or (
            isinstance(a, str) and isinstance(b, str) and a == b
        ):
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
    inner_a = shape_a[-1]
    inner_b = shape_b[-2]
    if isinstance(inner_a, int) and isinstance(inner_b, int) and inner_a != inner_b:
        raise ValueError(f"matmul inner-dim mismatch: {inner_a} vs {inner_b}")
    batch = _broadcast(shape_a[:-2], shape_b[:-2])
    if batch is None:
        raise ValueError(
            f"matmul batch dims are not broadcastable: {shape_a[:-2]} vs {shape_b[:-2]}"
        )
    out_shape = (*batch, shape_a[-2], shape_b[-1])
    return forge_tensor(list(out_shape), _dtype_str(a))


def _infer_linear_type(operand_types: list, forge_tensor):
    """``x @ weight.T + bias`` with weight shaped ``[out, in]``."""
    x = _expect_tensor(operand_types[0], forge_tensor)
    weight = _expect_tensor(operand_types[1], forge_tensor)
    x_shape = list(_shape_seq(x))
    w_shape = list(_shape_seq(weight))
    if len(x_shape) < 1 or len(w_shape) != 2:
        raise ValueError(f"linear needs x rank >= 1 and weight rank 2, got {x_shape}, {w_shape}")
    if x_shape[-1] != w_shape[-1]:
        raise ValueError(
            f"linear input width {x_shape[-1]} does not match weight width {w_shape[-1]}"
        )
    return forge_tensor([*x_shape[:-1], w_shape[0]], _dtype_str(x))


def _infer_broadcast_binary(operand_types: list, forge_tensor):
    a = _expect_tensor(operand_types[0], forge_tensor)
    b = _expect_tensor(operand_types[1], forge_tensor)
    out_shape = _broadcast(list(_shape_seq(a)), list(_shape_seq(b)))
    if out_shape is None:
        raise ValueError(
            f"binary op: shapes {_shape_seq(a)} and {_shape_seq(b)} are not broadcastable"
        )
    return forge_tensor(list(out_shape), _dtype_str(a))


def _infer_add_type(operand_types: list, forge_tensor):
    return _infer_broadcast_binary(operand_types, forge_tensor)


def _arg_name(arg) -> str:
    if isinstance(arg, Node):
        return arg.name
    raise TypeError(
        f"unexpected FX argument: {arg!r}; only Node placeholders are "
        "supported by the shipped FX importer (constants and Python ints "
        "are not yet lowered)."
    )


def _extract_reshape_target(node: Node, resolver, *, allow_symbolic: bool) -> list[int] | None:
    args = list(node.args)
    if len(args) < 2:
        return None
    target = args[1] if len(args) == 2 else list(args[1:])
    if not isinstance(target, (list, tuple)) and not isinstance(target, Node):
        return None
    try:
        return resolver(target)
    except NotImplementedError:
        if allow_symbolic:
            return None
        raise


def _flatten_output_args(args):
    """FX ``output`` nodes may carry a single Node or a tuple/list of them."""
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


def _build_op(spec: OpSpec, operands: list, result_type, node: Node | None = None):
    graph_mod = FxImporter._load_graph_dialect()
    tensor_mod = FxImporter._load_tensor_dialect()
    if spec.op == GEMM.op:
        return graph_mod.MatmulOp.create(operands=operands, result_types=[result_type])
    if spec.op == "forge.tensor.broadcast_add":
        return graph_mod.AddOp.create(operands=operands, result_types=[result_type])
    if spec.op == GELU.op:
        return graph_mod.GeluOp.create(operands=operands, result_types=[result_type])
    if spec.op == "forge.tensor.mul":
        return tensor_mod.MulOp.create(operands=operands, result_types=[result_type])
    if spec.op == "forge.tensor.relu":
        return tensor_mod.ReluOp.create(operands=operands, result_types=[result_type])
    if spec.op == "forge.tensor.layer_norm":
        return tensor_mod.LayerNormOp.create(
            operands=operands,
            result_types=[result_type],
            properties={
                "epsilon": tensor_mod.FloatAttr(
                    float(_extract_layer_norm_eps(node)), tensor_mod.Float32Type()
                ),
            },
        )
    if spec.op == "forge.tensor.softmax":
        return tensor_mod.SoftmaxOp.create(
            operands=operands,
            result_types=[result_type],
            properties={"dim": tensor_mod.IntAttr(int(_extract_softmax_dim(node)))},
        )
    if spec.op == "forge.tensor.scaled_dot_product_attention":
        return tensor_mod.ScaledDotProductAttentionOp.create(
            operands=operands,
            result_types=[result_type],
            properties={"num_heads": tensor_mod.IntAttr(1)},
        )
    raise AssertionError(f"_build_op: unknown forge op {spec.op!r}")  # pragma: no cover


def _extract_layer_norm_eps(node: Node | None) -> float:
    """Pull ``eps`` out of an ``aten.layer_norm`` FX node."""
    if node is None:
        return 1e-5
    args = list(getattr(node, "args", ()))
    if len(args) >= 5 and isinstance(args[4], (int, float)):
        return float(args[4])
    kwargs = getattr(node, "kwargs", {}) or {}
    eps = kwargs.get("eps")
    if isinstance(eps, (int, float)) and not isinstance(eps, bool):
        return float(eps)
    return 1e-5


def _extract_softmax_dim(node: Node | None) -> int:
    """Pull ``dim`` out of an ``aten.softmax`` FX node (args or kwargs)."""
    if node is None:
        return -1
    args = list(getattr(node, "args", ()))
    if len(args) >= 2 and isinstance(args[1], int) and not isinstance(args[1], bool):
        return args[1]
    kwargs = getattr(node, "kwargs", {}) or {}
    dim_v = kwargs.get("dim")
    if isinstance(dim_v, int) and not isinstance(dim_v, bool):
        return dim_v
    return -1
