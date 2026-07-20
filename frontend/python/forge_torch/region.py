"""Region boundary detection: walk FX graph in topological order and group into per-op regions (Phase 7 Wave 4)."""

# frontend/python/forge_torch/region.py
#
# Region boundary detection for the Phase 7 heterogeneous partitioner
# (Task 17, spec §6, plan §Todos).
#
# The partitioner (Task 18) consumes ``partition_into_regions`` to obtain a
# list of regions — each region is a list of FX nodes that share a single
# ``forge.tensor.*`` op name. Adjacent same-op nodes are merged into one
# region (the adjacent-merge heuristic), producing one region per conceptual
# op rather than one per individual FX node. This matches the spec §19
# "≈8-10 main ops" expectation for the transformer block.
#
# Skipped nodes:
#   * ``placeholder`` / ``output`` — handled by the FX importer separately.
#   * ``get_attr`` — handled by the FX importer separately.
#   * Pure-bookkeeping helpers — ``builtins.getattr``, ``operator.getitem``,
#     anything else with ``"getattr"`` or ``"getitem"`` in the target name.
#
# The returned ``List[List[FXNode]]`` is in topological order. Each inner
# list is itself a topologically-ordered region (FX preserves topology).

from __future__ import annotations

import torch
from torch.fx import GraphModule, Node

# ---------------------------------------------------------------------------
# call_function target → forge.tensor.* op name.
#
# Mirrors ``backend.py::_ATEN_TO_FORGE_TENSOR`` (Phase 7 capability check,
# Task 5) plus the Phase 7 ops the FX importer lowered in Task 14
# (layer_norm, softmax, scaled_dot_product_attention). Kept in sync with
# ``partitioner.py``'s copy so each module is usable standalone.
# ---------------------------------------------------------------------------

_ATEN_TO_FORGE_OP: dict[str, str] = {
    "aten.mm": "forge.tensor.gemm",
    "aten.matmul": "forge.tensor.gemm",
    "aten.addmm": "forge.tensor.gemm",
    "aten.bmm": "forge.tensor.gemm",
    "aten.add": "forge.tensor.broadcast_add",
    "aten.gelu": "forge.tensor.gelu",
    "aten.relu": "forge.tensor.gelu",
    "aten.softmax": "forge.tensor.softmax",
    "aten.layer_norm": "forge.tensor.layer_norm",
    "aten.view": "forge.tensor.reshape",
    "aten._unsafe_view": "forge.tensor.reshape",
    "aten.scaled_dot_product_attention": "forge.tensor.scaled_dot_product_attention",
}

# call_method target (string method name) → forge.tensor.* op name.
# The transformer block's manual ``_layer_norm`` lowers as a chain of
# call_method nodes (``mean`` / ``var`` / ``sub`` / ``sqrt`` / ``truediv``)
# plus one call_function ``add`` for the epsilon constant. Each method is
# mapped to the forge op it implements; the partitioner then handles the
# layered ops as one logical ``forge.tensor.layer_norm`` region via the
# adjacent-merge step below.

_METHOD_TO_FORGE_OP: dict[str, str] = {
    "mean": "forge.tensor.layer_norm",
    "var": "forge.tensor.layer_norm",
    "reshape": "forge.tensor.reshape",
    "view": "forge.tensor.reshape",
    "permute": "forge.tensor.reshape",
    "transpose": "forge.tensor.reshape",
    "matmul": "forge.tensor.gemm",
    "add": "forge.tensor.broadcast_add",
}


def _node_target_name(node: Node) -> str:
    """Best-effort reduction of ``node.target`` to ``aten.<op>`` form.

    ``torch.fx.symbolic_trace`` stores call_function targets as free-function
    objects (``operator.matmul``, ``torch._C._nn.gelu``); this helper extracts
    their ``__name__`` and applies the same aliases the FX importer uses
    (mirrors ``op_mapping.normalize_target``).
    """
    target = node.target
    name = getattr(target, "__name__", None)
    if not name:
        name = str(target)
    # Python builtins that map to aten.<op>.
    if name in {"matmul", "add", "mul"}:
        return f"aten.{name}"
    # C-extension aliases.
    if name in {"gelu", "relu", "silu", "softmax"}:
        return f"aten.{name}"
    if name.startswith("aten."):
        return name
    return f"aten.{name}"


def _is_pure_helper(node: Node) -> bool:
    """``True`` for ``builtins.getattr`` / ``operator.getitem`` / similar book-keeping ops."""
    target_str = str(node.target)
    return "getattr" in target_str or "getitem" in target_str


def map_node_to_forge_op(node: Node) -> str | None:
    """Map an FX node to a ``forge.tensor.*`` op name, or ``None``.

    Public helper so ``partitioner.py`` and downstream callers can use the
    same mapping logic. ``call_module`` nodes are mapped to
    ``forge.tensor.gemm`` (nn.Linear / linear-shaped submodule — the
    partitioner can further refine via the wrapped class when called with a
    richer registry).
    """
    if node.op in ("placeholder", "output", "get_attr"):
        return None
    if node.op == "call_module":
        return "forge.tensor.gemm"
    if node.op == "call_function":
        if _is_pure_helper(node):
            return None
        return _ATEN_TO_FORGE_OP.get(_node_target_name(node))
    if node.op == "call_method":
        if _is_pure_helper(node):
            return None
        return _METHOD_TO_FORGE_OP.get(str(node.target))
    return None


def partition_into_regions(
    model: torch.nn.Module,
    example_inputs: list[torch.Tensor],
) -> list[list[Node]]:
    """Partition the captured FX graph into regions.

    Walks ``torch.fx.symbolic_trace(model).graph`` in topological order.
    Adjacent nodes whose ``map_node_to_forge_op`` returns the same
    ``forge.tensor.*`` name are merged into one region. Nodes whose mapping
    is ``None`` (placeholders, outputs, get_attr, book-keeping helpers)
    are skipped — they never break or join a region.

    The returned ``List[List[FXNode]]`` preserves topological order across
    regions and within each region.
    """
    gm: GraphModule = torch.fx.symbolic_trace(model)
    regions: list[list[Node]] = []
    current_region: list[Node] = []
    current_op: str | None = None
    for node in gm.graph.nodes:
        mapped_op = map_node_to_forge_op(node)
        if mapped_op is None:
            continue
        if mapped_op != current_op:
            if current_region:
                regions.append(current_region)
            current_region = [node]
            current_op = mapped_op
        else:
            current_region.append(node)
    if current_region:
        regions.append(current_region)
    return regions


__all__ = (
    "map_node_to_forge_op",
    "partition_into_regions",
)
