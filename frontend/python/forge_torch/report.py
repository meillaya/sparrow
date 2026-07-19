"""Compilation observability report (per spec §16).

The :class:`CompilationReport` is the canonical payload every Phase-1
``torch.compile(backend="forgestack")`` call writes to
``.omo/evidence/phase1-report.txt``. Its sections mirror spec §16:

1. Model name
2. FX node count
3. MLIR op counts per pass (canonicalize etc.)
4. Fusion groups
5. Selected target
6. Alternative target costs (Astra, FPGA with transfer overhead)
7. Selected schedule (tile sizes, warps, stages)
8. Peak intermediate memory
9. Generated kernel count
10. Validation max error

The dataclass doubles as the JSON serialization target — ``to_json``
and :meth:`from_json` round-trip every field so callers can persist
the report alongside the cache artifact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Final, Protocol, TypedDict

MEBIBYTE: Final = 1024 * 1024


class AllocationView(Protocol):
    @property
    def byte_offset(self) -> int: ...

    @property
    def byte_size(self) -> int: ...


class ExecutionPlanView(Protocol):
    @property
    def allocations(self) -> Sequence[AllocationView]: ...


class Schedule(TypedDict, total=False):
    tile_m: int
    tile_n: int
    tile_k: int
    warps: int
    stages: int


class CompilationReportPayload(TypedDict):
    model_name: str
    captured_fx_nodes: int
    imported_mlir_operations: int
    operations_after_canonicalize: int
    fusion_groups: int
    selected_target: str
    alternative_astra_cost: str
    alternative_fpga_cost: str
    schedule: Schedule
    original_intermediate_memory_mb: float
    planned_intermediate_memory_mb: float
    generated_kernels: int
    validation_max_error: float


class PayloadLoader(Protocol):
    def __call__(self, s: str) -> CompilationReportPayload: ...


_LOAD_PAYLOAD: Final[PayloadLoader] = json.loads


@dataclass
class CompilationReport:
    """Spec §16 compilation observability report."""

    model_name: str
    captured_fx_nodes: int = 0
    imported_mlir_operations: int = 0
    operations_after_canonicalize: int = 0
    fusion_groups: int = 0
    selected_target: str = ""
    alternative_astra_cost: str = ""  # e.g. "1.31x"
    alternative_fpga_cost: str = ""  # e.g. "2.04x including transfer"
    schedule: Schedule = field(default_factory=Schedule)
    original_intermediate_memory_mb: float = 0.0
    planned_intermediate_memory_mb: float = 0.0
    generated_kernels: int = 0
    validation_max_error: float = 0.0
    plan: InitVar[ExecutionPlanView | None] = None

    def __post_init__(self, plan: ExecutionPlanView | None) -> None:
        if plan is None or not plan.allocations:
            return
        self.original_intermediate_memory_mb = (
            sum(allocation.byte_size for allocation in plan.allocations) / MEBIBYTE
        )
        self.planned_intermediate_memory_mb = (
            max(allocation.byte_offset + allocation.byte_size for allocation in plan.allocations)
            / MEBIBYTE
        )

    def render(self) -> str:
        """Render the report as a fixed-width human-readable string."""
        lines = [
            f"Model: {self.model_name}",
            "",
            f"Captured FX nodes:              {self.captured_fx_nodes}",
            f"Imported MLIR operations:       {self.imported_mlir_operations}",
            f"Operations after canonicalize:  {self.operations_after_canonicalize}",
            f"Fusion groups:                  {self.fusion_groups}",
            "",
            f"Selected target:                {self.selected_target}",
            f"Alternative Astra cost:         {self.alternative_astra_cost}",
            f"Alternative FPGA cost:          {self.alternative_fpga_cost}",
            "",
            "Selected schedule:",
            f"  tile M:                       {self.schedule.get('tile_m', '-')}",
            f"  tile N:                       {self.schedule.get('tile_n', '-')}",
            f"  tile K:                       {self.schedule.get('tile_k', '-')}",
            f"  warps:                        {self.schedule.get('warps', '-')}",
            f"  stages:                       {self.schedule.get('stages', '-')}",
            "",
            f"Original intermediate memory:   {self.original_intermediate_memory_mb:.1f} MB",
            f"Planned intermediate memory:     {self.planned_intermediate_memory_mb:.1f} MB",
            f"Generated kernels:               {self.generated_kernels}",
            f"Validation max error:            {self.validation_max_error:.4f}",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize all fields to a stable JSON string."""
        payload: CompilationReportPayload = {
            "model_name": self.model_name,
            "captured_fx_nodes": self.captured_fx_nodes,
            "imported_mlir_operations": self.imported_mlir_operations,
            "operations_after_canonicalize": self.operations_after_canonicalize,
            "fusion_groups": self.fusion_groups,
            "selected_target": self.selected_target,
            "alternative_astra_cost": self.alternative_astra_cost,
            "alternative_fpga_cost": self.alternative_fpga_cost,
            "schedule": self.schedule,
            "original_intermediate_memory_mb": self.original_intermediate_memory_mb,
            "planned_intermediate_memory_mb": self.planned_intermediate_memory_mb,
            "generated_kernels": self.generated_kernels,
            "validation_max_error": self.validation_max_error,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> CompilationReport:
        """Inverse of :meth:`to_json`; preserves every dataclass field."""
        payload = _LOAD_PAYLOAD(s)
        return cls(**payload)
