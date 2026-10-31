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
from collections.abc import Sequence
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
TRITON_TARGET = 0
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
class DeviceSpec:
    """One entry in the version 3 plan device table."""

    id: int
    name: str
    target: int
    memory_space: int = _HOST_SPACE


#: The CPU reference provider. Every plan needs one.
CPU_DEVICE = DeviceSpec(0, "cpu", CPU_TARGET)
#: The Triton-CPU fused kernel provider.
TRITON_DEVICE = DeviceSpec(1, "triton", TRITON_TARGET)


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


def _match_triton_group(op: object) -> dict[str, object] | None:
    """Match ``gemm → broadcast_add → gelu`` for the fused Triton kernel.

    The pattern must be a non-transposed 2D gemm, a rank-1 bias, and an
    exact GELU. Every intermediate value must have exactly one consumer so
    the CPU plan's tensor ids stay unambiguous.
    """
    if not isinstance(op, GeluOp) or bool(op.approximate):
        return None
    add = op.x.owner
    if not isinstance(add, BroadcastAddOp):
        return None
    if add.y.owner is not None:
        # A produced bias would need its own node; only block-argument or
        # parameter tensors are accepted here.
        pass
    gemm = add.x.owner
    if not isinstance(gemm, GemmOp):
        return None
    if bool(gemm.transpose_a) or bool(gemm.transpose_b):
        return None
    gemm_users = [use.operation for use in gemm.out.uses]
    if len(gemm_users) != 1 or gemm_users[0] is not add:
        return None
    add_users = [use.operation for use in add.out.uses]
    if len(add_users) != 1 or add_users[0] is not op:
        return None
    a_shape = _shape_of(gemm.a, what="triton gemm a")
    b_shape = _shape_of(gemm.b, what="triton gemm b")
    bias_shape = _shape_of(add.y, what="triton gemm bias")
    out_shape = _shape_of(op.out, what="triton gemm out")
    if len(a_shape) != 2 or len(b_shape) != 2 or len(bias_shape) != 1 or len(out_shape) != 2:
        return None
    if a_shape[1] != b_shape[0] or b_shape[1] != bias_shape[0]:
        return None
    return {"gemm": gemm, "add": add, "gelu": op}


def _triton_arguments(group: dict[str, object]) -> list[dict[str, Any]]:
    gemm = group["gemm"]
    add = group["add"]
    gelu = group["gelu"]
    assert isinstance(gemm, GemmOp) and isinstance(add, BroadcastAddOp)
    assert isinstance(gelu, GeluOp)
    specs = (
        ("x", gemm.a),
        ("w", gemm.b),
        ("b", add.y),
        ("y", gelu.out),
    )
    arguments: list[dict[str, Any]] = []
    for name, value in specs:
        arguments.append(
            {
                "name": name,
                "dtype": _dtype_of(value, what=f"triton {name}"),
                "is_input": name != "y",
                "shape": list(_shape_of(value, what=f"triton {name}")),
            }
        )
    return arguments


def build_plan(
    module: ModuleOp,
    *,
    name: str | None = None,
    devices: Sequence[DeviceSpec] | None = None,
    fuse_triton: bool = False,
) -> BuiltPlan:
    """Build a version 3 execution plan from a lowered module.

    The default device table is CPU-only. Passing a table that contains a
    Triton target and ``fuse_triton=True`` collapses a single
    ``gemm → broadcast_add → gelu`` chain into one Triton node, which is
    how a mixed CPU/Triton plan is produced. Ops that lack a CPU entry
    point raise :class:`UnsupportedOpError` (or the dialect verifier fails
    first).
    """
    func, block = _single_function(module)
    plan_name = name or str(func.sym_name.data)

    device_specs = list(devices) if devices is not None else [CPU_DEVICE]
    cpu_device = next((device for device in device_specs if device.target == CPU_TARGET), None)
    if cpu_device is None:
        raise PlanBuildError("plan needs a CPU device for its residual ops")
    triton_device = next(
        (device for device in device_specs if device.target == TRITON_TARGET), None
    )
    if fuse_triton and triton_device is None:
        triton_device = TRITON_DEVICE
        device_specs.append(triton_device)

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

    ops = [op for op in list(block.ops) if not isinstance(op, ReturnOp)]
    groups: dict[int, dict[str, object]] = {}
    skipped: set[int] = set()
    if triton_device is not None:
        for op in ops:
            group = _match_triton_group(op)
            if group is None:
                continue
            gemm = group["gemm"]
            assert isinstance(gemm, GemmOp)
            if id(gemm) in groups:
                continue
            groups[id(gemm)] = group
            skipped.update(id(item) for item in group.values())

    for op in ops:
        group = groups.get(id(op))
        if group is not None:
            gemm = group["gemm"]
            add = group["add"]
            gelu = group["gelu"]
            assert isinstance(gemm, GemmOp) and isinstance(add, BroadcastAddOp)
            assert isinstance(gelu, GeluOp)
            input_tensor_ids = []
            dependencies: set[int] = set()
            for operand in (gemm.a, gemm.b, add.y):
                if operand not in tensor_ids:
                    raise PlanBuildError(
                        "fused Triton group uses a value that is not a block "
                        "argument or an earlier op result"
                    )
                tensor_id = tensor_ids[operand]
                input_tensor_ids.append(tensor_id)
                if tensor_id in producer_of:
                    dependencies.add(producer_of[tensor_id])
            output_id = add_tensor(gelu.out, f"t{len(tensors)}")
            arguments = _triton_arguments(group)
            encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode()
            key = ("fused_gemm_bias_gelu", encoded)
            if key not in artifact_index:
                artifact_index[key] = len(artifacts)
                artifacts.append(
                    {
                        "target": TRITON_TARGET,
                        "entry_point": "fused_gemm_bias_gelu",
                        "binary": [],
                        "arguments": arguments,
                        "guards": [],
                        "workspace_bytes": 0,
                        "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
                    }
                )
            node_id = len(nodes)
            producer_of[output_id] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "device": triton_device.id,
                    "artifact": artifact_index[key],
                    "inputs": input_tensor_ids,
                    "outputs": [output_id],
                    "dependencies": sorted(dependencies),
                    "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
                }
            )
            continue
        if id(op) in skipped:
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

        input_tensor_ids = []
        dependencies = set()
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
                "device": cpu_device.id,
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
                    "device": cpu_device.id,
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
        "version": 3,
        "name": plan_name,
        "artifacts": artifacts,
        "devices": [
            {
                "id": device.id,
                "name": device.name,
                "target": device.target,
                "memory_space": device.memory_space,
            }
            for device in device_specs
        ],
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
    "CPU_DEVICE",
    "TRITON_DEVICE",
    "BuiltPlan",
    "DeviceSpec",
    "PlanBuildError",
    "TensorSpec",
    "UnsupportedDtypeError",
    "UnsupportedOpError",
    "UnsupportedShapeError",
    "build_plan",
)
