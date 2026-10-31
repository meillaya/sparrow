"""Config-aware compile entry point: CPU plan for A, raise for B and C."""

# frontend/python/forge_torch/compile_partitioned.py
#
# Config-aware compile entry point (spec §6, §19).
#
# ``compile_partitioned(model, example_inputs, *, config)`` captures the
# FX graph, partitions it into regions, and assigns each region a backend.
# Only config A is runnable today:
#
#   * ``"A"`` — every region -> ``cpu``. The callable is the CPU execution
#     plan, not the FX interpreter.
#   * ``"B"`` — elementwise -> Triton, matmul -> Astra. Raises
#     ``NotImplementedError`` until the Astra lowering exists.
#   * ``"C"`` — elementwise -> Triton, matmul -> MicroGPU. Raises until the
#     MicroGPU lowering exists.
#
# The region assignment stays data-driven from the canonical op table and
# is stored on the returned :class:`CompiledModel` for tooling and metrics.

from __future__ import annotations

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

      * ``"A"`` — always ``"cpu"``.
      * ``"B"`` — pure-matmul region → ``"astra"``; everything else →
        ``"triton"``.
      * ``"C"`` — matmul region → ``"microgpu"``; everything else →
        ``"triton"``.

    A region whose op set is empty or unknown defaults to ``"triton"``
    (the superset target).
    """
    if config == "A":
        return "cpu"

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

    Config A wraps the CPU execution-plan callable. The regions and
    per-region assignments are retained for introspection and for the
    spec §19 metrics table. Configs B and C are not constructed until the
    Astra and MicroGPU lowerings exist; the public entry point raises for
    them instead of pretending to compile.

    :attr:`execution` is ``"cpu"`` when the plan built, and
    :attr:`planning_only` mirrors whether execution still falls back to
    eager.
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
        compiled_callable=None,
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.regions = regions
        self.region_assignments = list(assignments)
        self.region_compiled = list(region_compiled)
        self.gm = gm
        self._compiled = compiled_callable
        self.execution = (
            getattr(compiled_callable, "execution", "eager")
            if compiled_callable is not None
            else "eager"
        )
        self.lowered = bool(getattr(compiled_callable, "lowered", False))
        self.planning_only = not self.lowered

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
    # Execution — dispatch to the CPU plan callable.
    # ------------------------------------------------------------------

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        if self._compiled is None:  # pragma: no cover - constructor guard
            raise RuntimeError("CompiledModel has no callable")
        out = self._compiled(*args)
        if isinstance(out, tuple):
            if len(out) == 1:
                return out[0]
            return out  # pragma: no cover - Phase 7 models are single-output
        return out


def compile_partitioned(
    model: torch.nn.Module,
    example_inputs: list[torch.Tensor],
    *,
    config: ValidConfig,
) -> CompiledModel:
    """Compile ``model`` for ``config`` ("A" | "B" | "C").

    Config A is the runnable path: it builds the CPU execution plan and
    returns a :class:`CompiledModel` whose callable runs that plan.
    Configs B and C raise :class:`NotImplementedError` naming the
    missing Astra / MicroGPU lowering; the region assignments remain
    available for planning and metrics.

    The model must be FX-traceable. Parameters may be ordinary module
    attributes; the importer lifts them into plan inputs.
    """
    if config not in {"A", "B", "C"}:
        raise ValueError(f"config must be one of 'A', 'B', 'C'; got {config!r}")
    if config != "A":
        backend = "Astra" if config == "B" else "MicroGPU"
        raise NotImplementedError(
            f"config {config!r} needs the {backend} lowering, which is not "
            "available; config A (CPU) is the only runnable configuration"
        )

    gm = torch.fx.symbolic_trace(model)
    regions = partition_into_regions(model, example_inputs)
    assignments = _assignments_from_config(regions, config)
    compiled = backend_compile(model, example_inputs, target="cpu")
    if not compiled.lowered:
        raise RuntimeError(
            "config 'A' did not produce a CPU plan: "
            f"{compiled.planning_error or 'unknown planning error'}"
        )
    return CompiledModel(
        model=model,
        example_inputs=example_inputs,
        regions=regions,
        assignments=assignments,
        region_compiled=[compiled],
        gm=gm,
        compiled_callable=compiled,
    )
