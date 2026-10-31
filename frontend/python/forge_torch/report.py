"""Compilation observability report (per spec §16).

The :class:`CompilationReport` records what the compiler measured and
what it could not measure. Measured fields carry values; fields that the
shipped path cannot populate carry ``None`` and render as ``unavailable``.
The earlier report filled several fields with constants for every model,
which made the report look complete when it was not.

The dataclass doubles as the JSON serialization target. ``to_json`` and
``from_json`` round-trip every field, including the optional ones.
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
    execution_path: str
    captured_fx_nodes: int
    imported_mlir_operations: int
    operations_after_canonicalize: int
    fusion_groups: int | None
    selected_target: str
    alternative_astra_cost: str | None
    alternative_fpga_cost: str | None
    schedule: Schedule
    original_intermediate_memory_mb: float | None
    planned_intermediate_memory_mb: float | None
    generated_kernels: int | None
    validation_max_error: float | None


class PayloadLoader(Protocol):
    def __call__(self, s: str) -> CompilationReportPayload: ...


_LOAD_PAYLOAD: Final[PayloadLoader] = json.loads


@dataclass
class CompilationReport:
    """Spec §16 compilation observability report.

    ``execution_path`` is ``eager`` for the shipped frontend path and
    ``native`` when a lowered callable exists. Optional fields stay
    ``None`` until the step that measures them runs.
    """

    model_name: str
    execution_path: str = "eager"
    captured_fx_nodes: int = 0
    imported_mlir_operations: int = 0
    operations_after_canonicalize: int = 0
    fusion_groups: int | None = None
    selected_target: str = ""
    alternative_astra_cost: str | None = None
    alternative_fpga_cost: str | None = None
    schedule: Schedule = field(default_factory=Schedule)
    original_intermediate_memory_mb: float | None = None
    planned_intermediate_memory_mb: float | None = None
    generated_kernels: int | None = None
    validation_max_error: float | None = None
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
            f"Execution path:                {self.execution_path}",
            "",
            f"Captured FX nodes:              {self.captured_fx_nodes}",
            f"Imported MLIR operations:       {self.imported_mlir_operations}",
            f"Operations after canonicalize:  {self.operations_after_canonicalize}",
            f"Fusion groups:                  {_format_int(self.fusion_groups)}",
            "",
            f"Selected target:                {self.selected_target}",
            f"Alternative Astra cost:         {_format_text(self.alternative_astra_cost)}",
            f"Alternative FPGA cost:          {_format_text(self.alternative_fpga_cost)}",
            "",
            "Selected schedule:",
            f"  tile M:                       {self.schedule.get('tile_m', '-')}",
            f"  tile N:                       {self.schedule.get('tile_n', '-')}",
            f"  tile K:                       {self.schedule.get('tile_k', '-')}",
            f"  warps:                        {self.schedule.get('warps', '-')}",
            f"  stages:                       {self.schedule.get('stages', '-')}",
            "",
            f"Original intermediate memory:   {_format_float(self.original_intermediate_memory_mb)}",
            f"Planned intermediate memory:     {_format_float(self.planned_intermediate_memory_mb)}",
            f"Generated kernels:               {_format_int(self.generated_kernels)}",
            f"Validation max error:            {_format_float(self.validation_max_error, digits=4)}",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize all fields to a stable JSON string."""
        payload: CompilationReportPayload = {
            "model_name": self.model_name,
            "execution_path": self.execution_path,
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


def _format_text(value: str | None) -> str:
    return value if value is not None else "unavailable"


def _format_int(value: int | None) -> str:
    return str(value) if value is not None else "unavailable"


def _format_float(value: float | None, *, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if value is not None else "unavailable"
