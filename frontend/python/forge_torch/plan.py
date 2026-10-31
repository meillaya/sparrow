"""Build a ForgeRT execution plan from lowered ``forge.tensor`` MLIR.

The plan builder is the bridge between the compiler and the runtime. It
walks one ``func.func`` in a verified module, assigns a tensor id to every
block argument and op result, and emits one plan node per tensor op.

The first backend is the CPU reference kernel registry. Every artifact
targets ``TargetKind::CPU`` and carries its kernel attributes as JSON in
``ArtifactDescriptor.binary``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import FloatAttr, IntAttr, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, SSAValue

from compiler.lib.Dialect.Graph import ForgeDType, TensorType
from compiler.lib.Dialect.Tensor import (
    BroadcastAddOp,
    FusedEpilogueOp,
    GeluOp,
    GemmOp,
    LayerNormOp,
    MulOp,
    ReluOp,
    ReshapeOp,
    ScaledDotProductAttentionOp,
    SelectOp,
    SoftmaxOp,
    TransposeOp,
)

CPU_TARGET = 4
F32_DTYPE = 0
_HOST_SPACE = 0


class PlanBuildError(Exception):
    """Base error for a module that cannot become an execution plan."""


class UnsupportedOpError(PlanBuildError):
    """The module contains an op the plan builder does not map."""


class UnsupportedDtypeError(PlanBuildError):
    """The general CPU path currently supports fp32 only."""


class UnsupportedShapeError(PlanBuildError):
    """The general CPU path currently supports static, non-empty shapes."""


@dataclass(frozen=True)
class TensorSpec:
    """One plan input or output."""

    id: int
    name: str
    shape: tuple[int, ...]
    dtype: int = F32_DTYPE


@dataclass(frozen=True)
class BuiltPlan:
    """A serialized plan plus the Python-side input and output specs."""

    plan: dict[str, Any]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    def to_json(self) -> str:
        return json.dumps(self.plan, sort_keys=True)


def _shape_of(value: SSAValue, *, what: str) -> tuple[int, ...]:
    if not isinstance(value.type, TensorType):
        raise UnsupportedShapeError(f"{what} must be !forge.tensor, got {value.type}")
    dims: list[int] = []
    for dim in value.type.get_shape():
        if not isinstance(dim, IntAttr):
            raise UnsupportedShapeError(
                f"{what} has a symbolic dimension ({dim}); the CPU plan requires static shapes"
            )
        dims.append(int(dim.data))
    if not dims:
        raise UnsupportedShapeError(f"{what} has rank 0; the CPU plan requires rank >= 1")
    return tuple(dims)


def _dtype_of(value: SSAValue, *, what: str) -> int:
    if not isinstance(value.type, TensorType):
        raise UnsupportedDtypeError(f"{what} must be !forge.tensor, got {value.type}")
    if value.type.get_dtype() is not ForgeDType.F32:
        raise UnsupportedDtypeError(
            f"{what} has dtype {value.type.get_dtype().value}; the CPU plan "
            "currently supports fp32 only"
        )
    return F32_DTYPE


def _tensor_spec(value: SSAValue, tensor_id: int, name: str) -> TensorSpec:
    return TensorSpec(
        id=tensor_id,
        name=name,
        shape=_shape_of(value, what=name),
        dtype=_dtype_of(value, what=name),
    )


def _artifact_for(op: object) -> tuple[str, dict[str, Any]]:
    """Return ``(entry_point, attributes)`` for one tensor op."""
    if isinstance(op, GemmOp):
        return "forge.tensor.gemm", {
            "transpose_a": bool(op.transpose_a),
            "transpose_b": bool(op.transpose_b),
        }
    if isinstance(op, FusedEpilogueOp):
        return "forge.tensor.fused_epilogue", {
            "activation": op.activation.data.value,
        }
    if isinstance(op, BroadcastAddOp):
        return "forge.tensor.broadcast_add", {}
    if isinstance(op, MulOp):
        return "forge.tensor.mul", {}
    if isinstance(op, ReluOp):
        return "forge.tensor.relu", {}
    if isinstance(op, TransposeOp):
        return "forge.tensor.transpose", {
            "permutation": list(op.permutation_dims()),
        }
    if isinstance(op, SelectOp):
        return "forge.tensor.select", {
            "dim": int(op.dim.data),
            "index": int(op.index.data),
        }
    if isinstance(op, GeluOp):
        return "forge.tensor.gelu", {
            "approximate": bool(op.approximate),
        }
    if isinstance(op, LayerNormOp):
        epsilon = op.epsilon
        if not isinstance(epsilon, FloatAttr):
            raise PlanBuildError("layer_norm epsilon must be a FloatAttr")
        raw_epsilon = getattr(epsilon, "value", epsilon)
        return "forge.tensor.layer_norm", {
            "epsilon": float(getattr(raw_epsilon, "data", raw_epsilon)),
        }
    if isinstance(op, SoftmaxOp):
        return "forge.tensor.softmax", {
            "dim": int(op.dim.data),
        }
    if isinstance(op, ScaledDotProductAttentionOp):
        return "forge.tensor.scaled_dot_product_attention", {
            "num_heads": int(op.num_heads.data),
        }
    if isinstance(op, ReshapeOp):
        return "forge.tensor.reshape", {}
    raise UnsupportedOpError(
        f"no CPU plan mapping for {op.name if hasattr(op, 'name') else type(op).__name__}"
    )


def _artifact_json(entry_point: str, attributes: dict[str, Any]) -> list[int]:
    encoded = json.dumps(attributes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return list(encoded)


def _argument_descriptors(op: object) -> list[dict[str, Any]]:
    arguments: list[dict[str, Any]] = []
    for index, operand in enumerate(op.operands):  # type: ignore[attr-defined]
        arguments.append(
            {
                "name": f"in{index}",
                "dtype": _dtype_of(operand, what=f"operand {index}"),
                "is_input": True,
                "shape": list(_shape_of(operand, what=f"operand {index}")),
            }
        )
    result = op.results[0]  # type: ignore[attr-defined]
    arguments.append(
        {
            "name": "out",
            "dtype": _dtype_of(result, what="result"),
            "is_input": False,
            "shape": list(_shape_of(result, what="result")),
        }
    )
    return arguments


def _single_function(module: ModuleOp) -> tuple[FuncOp, Block]:
    functions = [op for op in module.ops if isinstance(op, FuncOp)]
    if len(functions) != 1:
        raise PlanBuildError(f"expected exactly one func.func, found {len(functions)}")
    func = functions[0]
    blocks = list(func.body.blocks)
    if len(blocks) != 1:
        raise PlanBuildError(f"expected one block in {func.sym_name.data}, found {len(blocks)}")
    return func, blocks[0]


def build_plan(module: ModuleOp, *, name: str | None = None) -> BuiltPlan:
    """Build a CPU execution plan from a lowered module.

    The module must contain one ``func.func`` with one block and a
    ``func.return`` terminator. Every op must map to a CPU entry point.
    """
    func, block = _single_function(module)
    plan_name = name or str(func.sym_name.data)

    tensor_ids: dict[SSAValue, int] = {}
    tensors: list[TensorSpec] = []
    artifacts: list[dict[str, Any]] = []
    artifact_index: dict[tuple[str, bytes], int] = {}
    nodes: list[dict[str, Any]] = []
    producer_of: dict[int, int] = {}

    def add_tensor(value: SSAValue, label: str) -> int:
        tensor_id = len(tensors)
        spec = _tensor_spec(value, tensor_id, label)
        tensor_ids[value] = tensor_id
        tensors.append(spec)
        return tensor_id

    input_ids: list[int] = []
    for index, argument in enumerate(block.args):
        input_ids.append(add_tensor(argument, f"arg{index}"))

    for op in list(block.ops):
        if isinstance(op, ReturnOp):
            continue
        entry_point, attributes = _artifact_for(op)
        encoded = json.dumps(attributes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        key = (entry_point, encoded)
        if key not in artifact_index:
            artifact_index[key] = len(artifacts)
            artifacts.append(
                {
                    "target": CPU_TARGET,
                    "entry_point": entry_point,
                    "binary": _artifact_json(entry_point, attributes),
                    "arguments": _argument_descriptors(op),
                    "guards": [],
                    "workspace_bytes": 0,
                    "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
                }
            )
        artifact_id = artifact_index[key]

        input_tensor_ids: list[int] = []
        dependencies: set[int] = set()
        for operand in op.operands:
            if operand not in tensor_ids:
                raise PlanBuildError(
                    f"{op.name} uses a value that is not a block argument or an earlier op result"
                )
            tensor_id = tensor_ids[operand]
            input_tensor_ids.append(tensor_id)
            if tensor_id in producer_of:
                dependencies.add(producer_of[tensor_id])

        result = op.results[0]
        output_id = add_tensor(result, f"t{len(tensors)}")
        node_id = len(nodes)
        producer_of[output_id] = node_id
        nodes.append(
            {
                "id": node_id,
                "device": 0,
                "artifact": artifact_id,
                "inputs": input_tensor_ids,
                "outputs": [output_id],
                "dependencies": sorted(dependencies),
                "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
            }
        )

    terminator = block.last_op
    if not isinstance(terminator, ReturnOp):
        raise PlanBuildError("function must end with func.return")
    output_ids: list[int] = []
    for operand in terminator.operands:
        if operand not in tensor_ids:
            raise PlanBuildError("func.return uses a value with no tensor id")
        output_id = tensor_ids[operand]
        if output_id not in producer_of:
            # Pass-through input: copy it so every output has a producer.
            result_spec = next(spec for spec in tensors if spec.id == output_id)
            copy_artifact_key = ("forge.tensor.copy", b"")
            if copy_artifact_key not in artifact_index:
                artifact_index[copy_artifact_key] = len(artifacts)
                artifacts.append(
                    {
                        "target": CPU_TARGET,
                        "entry_point": "forge.tensor.copy",
                        "binary": [],
                        "arguments": [
                            {
                                "name": "in0",
                                "dtype": result_spec.dtype,
                                "is_input": True,
                                "shape": list(result_spec.shape),
                            },
                            {
                                "name": "out",
                                "dtype": result_spec.dtype,
                                "is_input": False,
                                "shape": list(result_spec.shape),
                            },
                        ],
                        "guards": [],
                        "workspace_bytes": 0,
                        "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
                    }
                )
            new_id = len(tensors)
            tensors.append(
                TensorSpec(id=new_id, name=f"out{len(output_ids)}", shape=result_spec.shape)
            )
            node_id = len(nodes)
            producer_of[new_id] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "device": 0,
                    "artifact": artifact_index[copy_artifact_key],
                    "inputs": [output_id],
                    "outputs": [new_id],
                    "dependencies": [],
                    "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
                }
            )
            output_id = new_id
        output_ids.append(output_id)

    plan = {
        "version": 2,
        "name": plan_name,
        "artifacts": artifacts,
        "tensors": [
            {
                "id": spec.id,
                "name": spec.name,
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "memory_space": _HOST_SPACE,
            }
            for spec in tensors
        ],
        "allocations": [],
        "nodes": nodes,
        "inputs": [
            {"tensor_id": tensor_id, "external_name": f"arg{index}"}
            for index, tensor_id in enumerate(input_ids)
        ],
        "outputs": [
            {"tensor_id": tensor_id, "external_name": f"out{index}"}
            for index, tensor_id in enumerate(output_ids)
        ],
    }

    inputs = tuple(tensor for tensor in tensors if tensor.id in set(input_ids))
    outputs = tuple(next(tensor for tensor in tensors if tensor.id == tid) for tid in output_ids)
    if not outputs:
        raise PlanBuildError("function returns no values")
    return BuiltPlan(plan=plan, inputs=inputs, outputs=outputs)


__all__ = (
    "BuiltPlan",
    "PlanBuildError",
    "TensorSpec",
    "UnsupportedDtypeError",
    "UnsupportedOpError",
    "UnsupportedShapeError",
    "build_plan",
)
