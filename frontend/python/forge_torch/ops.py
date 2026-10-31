"""Canonical op table shared by the frontend, partitioner, and plan builder.

Before this module the frontend carried four independent translations of
the same information: :mod:`forge_torch.op_mapping`, :mod:`forge_torch.region`,
:mod:`forge_torch.partitioner`, and the capability table in
:mod:`forge_torch.backend`. Adding an op meant editing all four and
hoping they stayed in sync. :data:`OP_SPECS` is now the single source of
truth; every other module derives its view from it.

Each :class:`OpSpec` records one canonical ``forge.tensor`` op plus the FX
spellings that lower to it. The optional ``graph_op`` names the legacy
``forge.graph`` op the importer used before the tensor dialect grew the
op, so the Phase 1 lowering path keeps working unchanged.

Adding an op
------------
1. Add a :class:`OpSpec` entry below.
2. Give the dialect an op (``compiler/lib/Dialect/Tensor``) or a lowering
   in the importer.
3. Add the CPU entry point to ``runtime/lib/providers/cpu_provider.cpp``.
4. Add the artifact mapping in ``forge_torch.plan``.
5. Cover the verifier, the C++ kernel, and the plan with tests.

The gates in ``docs/ROADMAP.md`` apply to every entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpSpec:
    """One canonical ``forge.tensor`` op and the FX forms that produce it.

    ``aten`` holds normalised ``aten.<name>`` targets (see
    :func:`normalize_target`); ``method`` holds ``call_method`` spellings
    such as ``transpose`` or ``reshape``. ``graph_op`` is set only for the
    ops the Phase 1 ``forge.graph`` dialect models directly. The cost
    shape is the fallback the partitioner uses when FX metadata carries no
    shape.
    """

    op: str
    aten: tuple[str, ...] = ()
    method: tuple[str, ...] = ()
    graph_op: str | None = None
    entry_point: str | None = None
    cost_shape: tuple[int, ...] = (8, 8)
    description: str = ""


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

GEMM = OpSpec(
    op="forge.tensor.gemm",
    aten=("aten.mm", "aten.matmul", "aten.bmm", "aten.addmm", "aten.linear"),
    method=("matmul", "mm", "bmm", "addmm"),
    graph_op="forge.graph.matmul",
    entry_point="forge.tensor.gemm",
    cost_shape=(8, 8, 8),
    description="matrix multiply, optionally with a bias epilogue",
)

BROADCAST_ADD = OpSpec(
    op="forge.tensor.broadcast_add",
    aten=("aten.add",),
    method=("add",),
    graph_op="forge.graph.add",
    entry_point="forge.tensor.broadcast_add",
    description="NumPy-broadcasting element-wise add",
)

MUL = OpSpec(
    op="forge.tensor.mul",
    aten=("aten.mul",),
    method=("mul",),
    entry_point="forge.tensor.mul",
    description="NumPy-broadcasting element-wise multiply",
)

RELU = OpSpec(
    op="forge.tensor.relu",
    aten=("aten.relu",),
    method=("relu",),
    entry_point="forge.tensor.relu",
    description="rectified linear unit",
)

GELU = OpSpec(
    op="forge.tensor.gelu",
    aten=("aten.gelu",),
    graph_op="forge.graph.gelu",
    entry_point="forge.tensor.gelu",
    description="Gaussian error linear unit",
)

RESHAPE = OpSpec(
    op="forge.tensor.reshape",
    aten=("aten.view", "aten._unsafe_view", "aten.reshape"),
    method=("reshape", "view"),
    graph_op="forge.graph.view",
    entry_point="forge.tensor.reshape",
    description="reshape with a contiguous copy at the runtime boundary",
)

TRANSPOSE = OpSpec(
    op="forge.tensor.transpose",
    aten=("aten.permute", "aten.transpose", "aten.t"),
    method=("permute", "transpose", "t"),
    entry_point="forge.tensor.transpose",
    description="permute the dimensions of a tensor",
)

SELECT = OpSpec(
    op="forge.tensor.select",
    aten=("aten.select", "aten.getitem"),
    entry_point="forge.tensor.select",
    description="take one index along a dimension and drop that dimension",
)

SOFTMAX = OpSpec(
    op="forge.tensor.softmax",
    aten=("aten.softmax",),
    entry_point="forge.tensor.softmax",
    description="softmax along one dimension",
)

LAYER_NORM = OpSpec(
    op="forge.tensor.layer_norm",
    aten=("aten.layer_norm",),
    entry_point="forge.tensor.layer_norm",
    description="layer normalisation over the last dimension",
)

SDPA = OpSpec(
    op="forge.tensor.scaled_dot_product_attention",
    aten=("aten.scaled_dot_product_attention",),
    entry_point="forge.tensor.scaled_dot_product_attention",
    cost_shape=(8, 8, 8),
    description="fused QK^T softmax V attention",
)

FUSED_EPILOGUE = OpSpec(
    op="forge.tensor.fused_epilogue",
    entry_point="forge.tensor.fused_epilogue",
    description="bias add and activation in one op",
)

COPY = OpSpec(
    op="forge.tensor.copy",
    entry_point="forge.tensor.copy",
    description="dense copy used for pass-through outputs",
)

OP_SPECS: Final[tuple[OpSpec, ...]] = (
    GEMM,
    BROADCAST_ADD,
    MUL,
    RELU,
    GELU,
    RESHAPE,
    TRANSPOSE,
    SELECT,
    SOFTMAX,
    LAYER_NORM,
    SDPA,
    FUSED_EPILOGUE,
    COPY,
)

#: ``aten.<op>`` spelling -> spec.
ATEN_OPS: Final[dict[str, OpSpec]] = {name: spec for spec in OP_SPECS for name in spec.aten}

#: ``call_method`` target spelling -> spec.
METHOD_OPS: Final[dict[str, OpSpec]] = {name: spec for spec in OP_SPECS for name in spec.method}

#: Every canonical ``forge.tensor`` op in the table.
TENSOR_OPS: Final[frozenset[str]] = frozenset(spec.op for spec in OP_SPECS)

#: CPU entry points the plan builder can map.
CPU_ENTRY_POINTS: Final[frozenset[str]] = frozenset(
    spec.entry_point for spec in OP_SPECS if spec.entry_point is not None
)

#: Ops the frontend recognises but does not lower yet. The importer raises
#: ``NotImplementedError`` naming the op when it meets one.
UNSUPPORTED_BUT_RECOGNISED: Final[frozenset[str]] = frozenset(
    {
        "aten.silu",
        "aten.sum",
        "aten.mean",
        "aten.baddbmm",
        "aten.div",
        "aten.pow",
    }
)

#: Backwards-compatible view: ``aten.<op>`` -> forge op the importer emits.
#: Prefer :func:`spec_for_aten` when the distinction between the graph and
#: tensor surfaces matters.
SUPPORTED_ATEN_OPS: Final[dict[str, str]] = {
    name: (spec.graph_op or spec.op) for name, spec in ATEN_OPS.items()
}

DTYPE_TO_FORGE: Final[dict[torch.dtype, str]] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int32: "i32",
    torch.int8: "i8",
    # i4 not yet representable as torch.dtype
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

#: Free-function spellings that reduce to an ``aten.<op>`` target.
_FUNCTION_ALIASES: Final[dict[str, str]] = {
    "matmul": "aten.matmul",
    "mm": "aten.mm",
    "bmm": "aten.bmm",
    "addmm": "aten.addmm",
    "linear": "aten.linear",
    "add": "aten.add",
    "mul": "aten.mul",
    "getitem": "aten.getitem",
    "truediv": "aten.div",
    "div": "aten.div",
    "reshape": "aten.reshape",
    "permute": "aten.permute",
    "transpose": "aten.transpose",
    "t": "aten.t",
    "gelu": "aten.gelu",
    "relu": "aten.relu",
    "silu": "aten.silu",
    "softmax": "aten.softmax",
    "layer_norm": "aten.layer_norm",
    "scaled_dot_product_attention": "aten.scaled_dot_product_attention",
    "sum": "aten.sum",
    "mean": "aten.mean",
    "pow": "aten.pow",
}


def normalize_target(target: object) -> str:
    """Reduce an FX ``node.target`` to its canonical ``aten.<op>`` spelling.

    ``torch.fx.symbolic_trace`` stores free functions (``operator.matmul``,
    ``torch._C._nn.gelu``); ``make_fx`` and Dynamo store ``OpOverload``
    instances (``torch.ops.aten.addmm.default``). Both forms, plus plain
    strings, reduce to the same key.
    """
    if isinstance(target, str):
        if target.startswith("aten.") and target.count(".") >= 2:
            namespace, op_name, _overload = target.split(".", 2)
            return f"{namespace}.{op_name}"
        return target
    raw = ""
    try:
        raw = str(target)
    except Exception:  # pragma: no cover - defensive
        raw = ""
    if raw.startswith("aten.") and raw.count(".") >= 2:
        namespace, op_name, _overload = raw.split(".", 2)
        return f"{namespace}.{op_name}"
    if raw.startswith("aten."):
        return raw
    name = getattr(target, "__name__", None) or getattr(target, "_name", None)
    if name is None:
        return ""
    if name.startswith("aten."):
        return name
    return _FUNCTION_ALIASES.get(name, f"aten.{name}")


def spec_for_aten(target: object) -> OpSpec:
    """Return the :class:`OpSpec` for an FX ``call_function`` target.

    Raises :class:`NotImplementedError` for recognised-but-unlowered ops
    and :class:`KeyError` for targets the frontend has never heard of.
    """
    normalized = normalize_target(target)
    spec = ATEN_OPS.get(normalized)
    if spec is not None:
        return spec
    if normalized in UNSUPPORTED_BUT_RECOGNISED:
        raise NotImplementedError(
            f"{normalized!s} is recognised by the Sparrow frontend but is "
            "not yet lowered by the shipped FX importer."
        )
    raise KeyError(f"unknown FX op target: {target!r} (normalised to {normalized!r})")


def spec_for_method(name: str) -> OpSpec:
    """Return the :class:`OpSpec` for a ``call_method`` target, or raise."""
    spec = METHOD_OPS.get(name)
    if spec is not None:
        return spec
    raise KeyError(f"unknown FX method target: {name!r}")


def map_node_to_forge(target: object) -> str:
    """Return the forge op name for an FX target (legacy helper)."""
    spec = spec_for_aten(target)
    return spec.graph_op or spec.op


def is_shape_metadata_node(node: object) -> bool:
    """Return ``True`` for FX nodes that compute shapes, not tensors.

    ``symbolic_trace`` turns ``x.shape[0]`` into a chain of ``getattr`` /
    ``getitem`` nodes. They have no tensor SSA value and must not become
    partitioner regions or capability-check ops.
    """
    op = getattr(node, "op", None)
    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    if op == "call_function":
        if "getattr" in target:
            return len(args) >= 2 and args[1] == "shape"
        if "getitem" in target:
            return bool(args) and is_shape_metadata_node(args[0])
        return False
    return op == "call_method" and target == "size"


def cost_shape_for(op: str) -> list[int]:
    """Return the partitioner's fallback cost shape for ``op``."""
    for spec in OP_SPECS:
        if spec.op == op:
            return list(spec.cost_shape)
    return [1, 1]


__all__ = (
    "ATEN_OPS",
    "CPU_ENTRY_POINTS",
    "DTYPE_TO_FORGE",
    "METHOD_OPS",
    "OP_SPECS",
    "SUPPORTED_ATEN_OPS",
    "TENSOR_OPS",
    "UNSUPPORTED_BUT_RECOGNISED",
    "OpSpec",
    "cost_shape_for",
    "is_shape_metadata_node",
    "map_node_to_forge",
    "normalize_target",
    "spec_for_aten",
    "spec_for_method",
)
