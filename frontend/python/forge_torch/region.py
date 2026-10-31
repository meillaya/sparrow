"""Region boundary detection: group FX nodes into per-op regions (Phase 7)."""

# frontend/python/forge_torch/region.py
#
# Region boundary detection for the heterogeneous partitioner. The
# partitioner consumes ``partition_into_regions`` to obtain a list of
# regions — each region is a list of FX nodes that share a single
# ``forge.tensor.*`` op name. Adjacent same-op nodes are merged into one
# region (the adjacent-merge heuristic), producing one region per
# conceptual op rather than one per individual FX node.
#
# The op -> region-name translation lives in :mod:`forge_torch.ops`.
# This module no longer keeps a private mapping table.
#
# Skipped nodes:
#   * ``placeholder`` / ``output`` / ``get_attr`` — handled separately.
#   * Pure-bookkeeping helpers — ``getattr`` and size/index nodes.

from __future__ import annotations

import torch
from torch.fx import GraphModule, Node

from forge_torch.ops import (
    GEMM,
    LAYER_NORM,
    OpSpec,
    is_shape_metadata_node,
    spec_for_aten,
    spec_for_method,
)


def _spec_or_none(node: Node) -> OpSpec | None:
    if node.op == "call_function":
        target = str(node.target)
        if "getattr" in target and "shape" in target:
            return None
        try:
            return spec_for_aten(node.target)
        except (KeyError, NotImplementedError):
            return None
    if node.op == "call_method":
        name = str(node.target)
        if name in {"size", "dim", "numel", "getitem"}:
            return None
        try:
            return spec_for_method(name)
        except KeyError:
            return None
    return None


def map_node_to_forge_op(node: Node) -> str | None:
    """Map an FX node to a ``forge.tensor.*`` op name, or ``None``.

    Public helper so ``partitioner.py`` and downstream callers can use the
    same mapping logic. ``call_module`` nodes resolve by module class:
    ``nn.Linear`` is a gemm and ``nn.LayerNorm`` a layer norm.
    """
    if node.op in ("placeholder", "output", "get_attr"):
        return None
    if is_shape_metadata_node(node):
        return None
    if node.op == "call_module":
        target = str(node.target)
        # ``nn.Linear`` submodules carry a weight and bias under the usual
        # names; ``nn.LayerNorm`` carries a normalized-shape attribute.
        owning = getattr(node.graph, "owning_module", None)
        submodule = getattr(owning, target, None) if owning is not None else None
        if isinstance(submodule, torch.nn.LayerNorm):  # type: ignore[arg-type]
            return LAYER_NORM.op
        if isinstance(submodule, torch.nn.Linear):  # type: ignore[arg-type]
            return GEMM.op
        return GEMM.op
    if node.op == "call_function":
        if str(node.target) in {"builtins.getattr", "operator.getattr"}:
            return None
        spec = _spec_or_none(node)
        return spec.op if spec is not None else None
    if node.op == "call_method":
        spec = _spec_or_none(node)
        return spec.op if spec is not None else None
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
