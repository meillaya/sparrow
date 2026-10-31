"""Greedy partitioner per spec §6: pick cheapest valid backend per region subject to capabilities (Phase 7 Wave 4)."""

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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final

from torch.fx import Node

from forge_torch.capabilities import BackendCapabilities
from forge_torch.cost_estimator import CostEstimate, estimate_cost
from forge_torch.lowerings import COMPILER_LOWERINGS

# ---------------------------------------------------------------------------
# Per-op shape defaults for cost estimation.
#
# When an FX node does not carry ``meta['tensor_meta']`` or
# ``meta['cost_shape']``, the partitioner falls back to a per-op default.
# The defaults match the ``cost_estimator`` shape contract:
#
#   * matmul / gemm / mm / addmm -> ``[M, N, K]`` with small values so the
#     Triton vs Astra ranking is sensitive (M=2..8 keeps Triton competitive
#     when astra's per-tile overhead dominates small shapes).
#   * elementwise / reductions / reshape -> ``[M, N]``.
#   * matmul-shaped sdpa -> ``[M, N, K]``.
# ---------------------------------------------------------------------------

_DEFAULT_COST_SHAPE: Final[dict[str, list[int]]] = {
    "forge.tensor.gemm": [8, 8, 8],
    "forge.tensor.fused_epilogue": [8, 8],
    "forge.tensor.broadcast_add": [8, 8],
    "forge.tensor.gelu": [8, 8],
    "forge.tensor.layer_norm": [8, 8],
    "forge.tensor.softmax": [8, 8],
    "forge.tensor.scaled_dot_product_attention": [8, 8, 8],
    "forge.tensor.reshape": [8, 8],
}

# call_function target → forge.tensor.* op name.
# (Mirrors ``region.py``'s copy so each module is usable standalone.)
_ATEN_TO_FORGE_OP: Final[dict[str, str]] = {
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

# call_method → forge.tensor.* op name.
_METHOD_TO_FORGE_OP: Final[dict[str, str]] = {
    "mean": "forge.tensor.layer_norm",
    "var": "forge.tensor.layer_norm",
    "reshape": "forge.tensor.reshape",
    "view": "forge.tensor.reshape",
    "permute": "forge.tensor.reshape",
    "transpose": "forge.tensor.reshape",
    "matmul": "forge.tensor.gemm",
    "add": "forge.tensor.broadcast_add",
}

# Builtins alias table (mirrors ``op_mapping.normalize_target``).
_BUILTIN_TO_ATEN: Final[dict[str, str]] = {
    "matmul": "aten.matmul",
    "add": "aten.add",
    "mul": "aten.mul",
    "gelu": "aten.gelu",
    "relu": "aten.relu",
    "silu": "aten.silu",
    "softmax": "aten.softmax",
}


def _node_target_aten_name(node: Node) -> str:
    """Best-effort reduction of ``node.target`` to ``aten.<op>`` form."""
    target = node.target
    name = getattr(target, "__name__", None)
    if not name:
        name = str(target)
    if name in _BUILTIN_TO_ATEN:
        return _BUILTIN_TO_ATEN[name]
    if name.startswith("aten."):
        return name
    return f"aten.{name}"


def map_node_to_forge_op(node: Node) -> str | None:
    """Map an FX node to its ``forge.tensor.*`` op name, or ``None``.

    Same heuristic as :func:`forge_torch.region.map_node_to_forge_op` —
    see that module's docstring for the rationale. Kept as a separate copy
    here so the partitioner can be imported without pulling in the region
    detector.
    """
    if node.op in ("placeholder", "output", "get_attr"):
        return None
    if node.op == "call_module":
        # nn.Linear / linear-shaped submodules → ``forge.tensor.gemm``.
        return "forge.tensor.gemm"
    target_str = str(node.target)
    if "getattr" in target_str or "getitem" in target_str:
        return None
    if node.op == "call_function":
        return _ATEN_TO_FORGE_OP.get(_node_target_aten_name(node))
    if node.op == "call_method":
        return _METHOD_TO_FORGE_OP.get(target_str)
    return None


def _node_cost_shape(node: Node, op_name: str) -> list[int]:
    """Extract the per-op cost shape for ``node``.

    Priority order:

    1. ``node.meta['cost_shape']`` (test-only override; lets pytest pin
       deterministic shapes for the cycle count).
    2. ``node.meta['tensor_meta']['shape']`` or
       ``node.meta['tensor_meta'].shape`` (the standard FX metadata set by
       ``make_fx``).
    3. Per-op default in :data:`_DEFAULT_COST_SHAPE`.
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
    return list(_DEFAULT_COST_SHAPE.get(op_name, [1, 1]))


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
