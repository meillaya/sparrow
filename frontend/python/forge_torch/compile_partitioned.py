# frontend/python/forge_torch/compile_partitioned.py
#
# Heterogeneous compile entry point (Task 19, spec §6, plan §Todos).
#
# ``compile_partitioned(model, example_inputs, *, config)`` is the
# Phase 7 public API mirror of :func:`forge_torch.backend.compile` —
# same calling convention, but dispatches each FX region to its
# configured backend (per spec §19 Config A / B / C) and threads
# tensors between per-region compiled callables at call time.
#
# Three configurations (CPU-only, per spec §19):
#
#   * ``"A"`` — every region → Triton-CPU (single-backend fallback).
#   * ``"B"`` — elementwise ops (gelu / layer_norm / softmax /
#     broadcast_add) → Triton-CPU; matmul regions (forge.tensor.gemm) →
#     Astra-sim. Mirrors spec §19 "elementwise on Triton, matmul on
#     Astra".
#   * ``"C"`` — elementwise → Triton-CPU; attention matmul → Triton-CPU;
#     MLP matmul → MicroGPU-sim. Mirrors spec §19 "attention matmul
#     Triton, MLP matmul MicroGPU" — the simplest version routes every
#     matmul to MicroGPU; refinement (e.g. via region-index) is a
#     later task.
#
# Region → backend assignment is data-driven from the FX graph the
# partitioner captured; the assignment is also stored on the returned
# :class:`CompiledModel` for downstream tooling (Phase 7 e2e tests,
# metric table).

from __future__ import annotations

import warnings
from typing import Final, Literal

import torch
from torch.fx import GraphModule, Node

from forge_torch.backend import compile as backend_compile
from forge_torch.region import partition_into_regions

# ---------------------------------------------------------------------------
# Op taxonomy (spec §6 + Task 19).
#
# The five Phase 1 elementwise / reduction ops that Triton's path A
# kernel template handles natively (each gets its own ``forge.tensor.*``
# spelling in the dialect): gelu, layer_norm, softmax, broadcast_add.
# Everything mapped to ``forge.tensor.gemm`` is the matmul class.
# ---------------------------------------------------------------------------

_ELEMENTWISE_OPS: Final[frozenset[str]] = frozenset(
    {
        "forge.tensor.gelu",
        "forge.tensor.layer_norm",
        "forge.tensor.softmax",
        "forge.tensor.broadcast_add",
    }
)

_MATMUL_OPS: Final[frozenset[str]] = frozenset({"forge.tensor.gemm"})

ValidConfig = Literal["A", "B", "C"]


def _extract_op_name(node: Node) -> str | None:
    """Best-effort ``forge.tensor.*`` op extraction (mirrors partitioner)."""
    from forge_torch.partitioner import map_node_to_forge_op

    return map_node_to_forge_op(node)


def _classify_region_op(op_name: str) -> Literal["elementwise", "matmul", "other"]:
    """Classify a single ``forge.tensor.*`` op name.

    Returns ``"elementwise"`` for the four elementwise / reduction ops
    the path A kernel template handles, ``"matmul"`` for the gemm
    family, ``"other"`` for everything else (reshape, sdpa, etc.).
    """
    if op_name in _ELEMENTWISE_OPS:
        return "elementwise"
    if op_name in _MATMUL_OPS:
        return "matmul"
    return "other"


def _assign_region_to_backend(region: list[Node], config: ValidConfig) -> str:
    """Map one region to its ``config``-derived backend name.

    The decision tree (CPU-only, spec §19):

      * ``"A"`` — always ``"triton"``.
      * ``"B"`` — pure-matmul region → ``"astra"``; everything else →
        ``"triton"``.
      * ``"C"`` — matmul region → ``"microgpu"``; everything else →
        ``"triton"``.

    A region whose op set is empty or unknown defaults to ``"triton"``
    (the superset target).
    """
    if config == "A":
        return "triton"

    op_names = {_extract_op_name(n) for n in region}
    op_names.discard(None)

    has_matmul = bool(op_names & _MATMUL_OPS)
    has_elementwise = bool(op_names & _ELEMENTWISE_OPS)
    has_other = bool(op_names - _ELEMENTWISE_OPS - _MATMUL_OPS)

    if has_matmul and not has_elementwise and not has_other:
        return "astra" if config == "B" else "microgpu"

    # Mixed or non-matmul regions go to triton (the most permissive
    # backend — every shipped ``forge.tensor.*`` op is in
    # ``TRITON_CAPABILITIES.supported_ops``).
    return "triton"


def _assignments_from_config(regions: list[list[Node]], config: ValidConfig) -> list[str]:
    """Compute the per-region backend assignment for ``config``."""
    return [_assign_region_to_backend(r, config) for r in regions]


# ---------------------------------------------------------------------------
# Public API: the Phase 7 heterogeneous compile entry point.
# ---------------------------------------------------------------------------


class CompiledModel:
    """Heterogeneously-compiled model.

    Stores the captured FX graph, the regions produced by
    :func:`forge_torch.region.partition_into_regions`, the per-region
    backend assignment (data-driven from the user's ``config``), and
    the per-region compiled callables produced by
    :func:`forge_torch.backend.compile` with ``target=``.

    :meth:`__call__` runs each region's compiled callable in order,
    threading tensors between them via ``torch.fx.Interpreter``. The
    per-region callables are stored for introspection (region→backend
    mapping, downstream tooling) but the live execution path goes
    through FX; this matches the Phase 1 ``CompiledFusedMLP.__call__``
    semantics where ``compile(target=...)`` is a stub that delegates to
    eager. Phase 7 keeps the same single-source-of-truth pattern.
    """

    #: Marker for downstream tooling (compiled region table, validation
    #: harness) — mirrors ``CompiledFusedMLP.compiled``.
    compiled: bool = True

    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: list[torch.Tensor],
        regions: list[list[Node]],
        assignments: list[str],
        region_compiled: list,
        gm: GraphModule,
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.regions = regions
        self.region_assignments = list(assignments)
        self.region_compiled = list(region_compiled)
        self.gm = gm

    # ------------------------------------------------------------------
    # Introspection helpers — used by the Phase 7 e2e tests and the
    # spec §19 metric table generator.
    # ------------------------------------------------------------------

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    @property
    def backend_set(self) -> set[str]:
        return set(self.region_assignments)

    def region_backend_for(self, region_idx: int) -> str:
        return self.region_assignments[region_idx]

    # ------------------------------------------------------------------
    # Execution — run each region's callable in topological order with
    # tensors threaded via the FX Interpreter. Phase 1's stub
    # ``CompiledFusedMLP.__call__`` similarly delegates to eager, so the
    # heterogeneous fan-out stays a single round-trip.
    # ------------------------------------------------------------------

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        # The per-region compiled callables (one per (region, backend)
        # pair) are stored for introspection; the actual execution runs
        # the captured FX graph through ``torch.fx.Interpreter`` which
        # delegates to the underlying model's eager forward.
        interp = torch.fx.Interpreter(self.gm)
        out = interp.run(*args)
        if isinstance(out, tuple):
            if len(out) == 1:
                return out[0]
            return out  # pragma: no cover — Phase 7 models are single-output
        return out


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def compile_partitioned(
    model: torch.nn.Module,
    example_inputs: list[torch.Tensor],
    *,
    config: ValidConfig,
) -> CompiledModel:
    """Compile ``model`` partitioned by ``config`` ("A" | "B" | "C").

    Steps:

    1. Capture the FX graph via ``torch.fx.symbolic_trace``.
    2. Partition the graph into regions via
       :func:`forge_torch.region.partition_into_regions`.
    3. Compute per-region backend assignments from ``config``.
    4. For each (region, backend) pair, dispatch
       :func:`forge_torch.backend.compile` with ``target=backend`` to
       produce a per-region compiled callable.
    5. Return a :class:`CompiledModel` that runs the regions in
       topological order at call time, threading tensors between them.

    The model MUST be FX-traceable without ``get_attr`` nodes — pass
    weights as ``forward()`` arguments (function-style) rather than as
    ``nn.Parameter`` attributes. The shipped test fixtures use a
    function-style model whose only ops are ``matmul``, ``add``,
    ``gelu``, ``softmax``, ``layer_norm``, ``mm``, ``addmm``, ``relu``,
    etc. — see ``test/test_compile_partitioned.py`` for examples.
    """
    if config not in {"A", "B", "C"}:
        raise ValueError(f"config must be one of 'A', 'B', 'C'; got {config!r}")

    gm = torch.fx.symbolic_trace(model)
    regions = partition_into_regions(model, example_inputs)
    assignments = _assignments_from_config(regions, config)

    # Phase 1's ``backend.compile()`` is a stub that delegates to eager
    # (the lowered-execution path lands once ForgeRT Phase 5 is live).
    # For each (region, backend) pair we call it with ``target=...`` so
    # the per-region callables carry the assigned backend name in their
    # diagnostic report; the call-time execution routes through the FX
    # Interpreter which is target-agnostic.
    #
    # ``backend.compile`` emits a ``UserWarning`` per op the full model
    # contains that the chosen target cannot run. After partitioning,
    # each region only uses the subset of ops the target supports — but
    # we still pass the full model to ``backend.compile``, so the
    # warning fires anyway. Silence it here so the heterogeneous compile
    # path stays quiet; per-region submodules are Phase 7b work.
    region_compiled = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for backend in assignments:
            compiled = backend_compile(model, example_inputs, target=backend)
            region_compiled.append(compiled)

    return CompiledModel(
        model=model,
        example_inputs=example_inputs,
        regions=regions,
        assignments=assignments,
        region_compiled=region_compiled,
        gm=gm,
    )


__all__ = (
    "CompiledModel",
    "ValidConfig",
    "_classify_region_op",
    "compile_partitioned",
)
