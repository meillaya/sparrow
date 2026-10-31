"""Greedy partitioner per spec §6: cheapest valid backend per region (Phase 7)."""

# frontend/python/forge_torch/partitioner.py
#
# Greedy heterogeneous partitioner (Task 18, spec §6, plan §Todos).
#
# ``partition(regions, capabilities, cost_estimator)`` selects the cheapest
# valid backend for each region subject to capability coverage. The
# algorithm (spec §6 "greedy first, DP later"):
#
#   1. For every region: extract the set of ``forge.tensor.*`` op names
#      via :func:`map_node_to_forge_op` (same mapping as ``region.py``).
#   2. Filter candidate backends: a backend is valid iff every region op
#      is in ``BackendCapabilities.supported_ops``. If a region's op isn't
#      in any backend's supported_ops, partitioner falls back to
#      ``"triton"`` (the superset target — every shipped op is in
#      ``TRITON_CAPABILITIES``).
#   3. Score each candidate by ``cost_estimator(region, backend_name)
#      .compute_cycles``.
#   4. Sort by ``(cycles, backend_name)``; the lowest wins. Tiebreak is
#      alphabetical order on the backend name (deterministic + friendly
#      to golden tests).
#
# Spec §6 explicitly defers a DP partitioner to Phase 7b; the greedy form
# is sufficient for the small Phase 7 transformer (~10 regions).
#
# The FX-target mapping lives in :mod:`forge_torch.ops`.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch.fx import Node

from forge_torch.capabilities import BackendCapabilities
from forge_torch.cost_estimator import CostEstimate, estimate_cost
from forge_torch.lowerings import COMPILER_LOWERINGS
from forge_torch.ops import (
    GEMM,
    LAYER_NORM,
    cost_shape_for,
    is_shape_metadata_node,
    spec_for_aten,
    spec_for_method,
)

# call_function target -> forge.tensor.* op name, derived from the
# canonical table. ``None`` entries are recognised helpers, not tensor ops.


def map_node_to_forge_op(node: Node) -> str | None:
    """Map an FX node to its ``forge.tensor.*`` op name, or ``None``.

    Same logic as :func:`forge_torch.region.map_node_to_forge_op`; the
    op names come from :mod:`forge_torch.ops`.
    """
    if node.op in ("placeholder", "output", "get_attr"):
        return None
    if is_shape_metadata_node(node):
        return None
    if node.op == "call_module":
        owning = getattr(node.graph, "owning_module", None)
        submodule = getattr(owning, str(node.target), None) if owning is not None else None
        if isinstance(submodule, torch.nn.LayerNorm):  # type: ignore[arg-type]
            return LAYER_NORM.op
        return GEMM.op
    target_str = str(node.target)
    if "getattr" in target_str:
        return None
    if node.op == "call_function":
        try:
            return spec_for_aten(node.target).op
        except (KeyError, NotImplementedError):
            return None
    if node.op == "call_method":
        try:
            return spec_for_method(target_str).op
        except KeyError:
            return None
    return None


def _node_cost_shape(node: Node, op_name: str) -> list[int]:
    """Extract the per-op cost shape for ``node``.

    Priority order:

    1. ``node.meta['cost_shape']`` (test-only override; lets pytest pin
       deterministic shapes for the cycle count).
    2. ``node.meta['tensor_meta']['shape']`` or
       ``node.meta['tensor_meta'].shape`` (the standard FX metadata set by
       ``make_fx``).
    3. Per-op default from :attr:`forge_torch.ops.OpSpec.cost_shape`.
    """
    meta = getattr(node, "meta", None) or {}
    explicit = meta.get("cost_shape")
    if explicit is not None:
        return list(explicit)
    tensor_meta = meta.get("tensor_meta")
    if tensor_meta is not None:
        if isinstance(tensor_meta, dict):
            shape = tensor_meta.get("shape")
            if shape is not None:
                return list(shape)
        else:
            shape = getattr(tensor_meta, "shape", None)
            if shape is not None:
                return list(shape)
    return cost_shape_for(op_name)


class UnsupportedRegionError(ValueError):
    """Raised when no backend can lower every op in a region."""


def partition(
    regions: Sequence[Sequence[Node]],
    capabilities: dict[str, BackendCapabilities],
    cost_estimator: Callable[[Sequence[dict], str], CostEstimate] = estimate_cost,
    *,
    lowerings: Mapping[str, frozenset[str]] | None = None,
) -> list[tuple[list[Node], str]]:
    """Pick the cheapest valid backend per region (spec §6 greedy).

    ``regions`` is a sequence of regions; each region is a sequence of FX
    nodes. ``capabilities`` maps ``backend_name`` to its
    :class:`BackendCapabilities` (use :data:`forge_torch.capabilities.BACKEND_CAPABILITIES`
    to get the shipped three). ``cost_estimator`` defaults to
    :func:`forge_torch.cost_estimator.estimate_cost` but can be overridden
    in tests.

    ``lowerings`` maps a backend name to the ops the compiler can emit.
    It defaults to :data:`forge_torch.lowerings.COMPILER_LOWERINGS`, which
    is empty until a region codegen lands. A caller that is testing the
    selection algorithm can pass a wider map.

    Returns a ``list[tuple[list[FXNode], str]]`` parallel to ``regions``:
    each element is ``(region_nodes, chosen_backend_name)``. Raises
    :class:`UnsupportedRegionError` when a region has no mappable op or
    no backend covers every op in both the capability and lowering sets.
    """
    lowering_map = COMPILER_LOWERINGS if lowerings is None else lowerings
    result: list[tuple[list[Node], str]] = []
    for region in regions:
        node_list = list(region)
        region_dicts: list[dict] = []
        region_op_names: set[str] = set()
        for node in node_list:
            op_name = map_node_to_forge_op(node)
            if op_name is None:
                continue
            region_op_names.add(op_name)
            region_dicts.append({"op": op_name, "shape": _node_cost_shape(node, op_name)})

        if not region_op_names:
            raise UnsupportedRegionError(
                "region has no mappable forge.tensor op; cannot choose a backend"
            )

        # Score each candidate backend. A backend must cover the op set in
        # both what it can run and what the compiler can emit.
        candidates: list[tuple[int, str]] = []
        for backend_name, caps in capabilities.items():
            emitted = lowering_map.get(backend_name, frozenset())
            if not region_op_names.issubset(caps.supported_ops & emitted):
                continue
            if not region_dicts:
                # Region has op names but no cost dicts (e.g. all nodes
                # had None mapping — shouldn't happen given the guard above,
                # but be defensive): skip cost estimation, give it 0 cost.
                candidates.append((0, backend_name))
                continue
            cost = cost_estimator(region_dicts, backend_name)
            candidates.append((cost.compute_cycles, backend_name))

        if not candidates:
            raise UnsupportedRegionError(
                f"no backend covers region ops {sorted(region_op_names)} with a compiler lowering"
            )

        # Sort by (cycles, backend_name); lowest wins. The tuple sort is
        # stable and the backend_name tiebreak is alphabetical per spec §6.
        candidates.sort(key=lambda c: (c[0], c[1]))
        result.append((node_list, candidates[0][1]))

    return result


__all__ = (
    "UnsupportedRegionError",
    "map_node_to_forge_op",
    "partition",
)
