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
from dataclasses import asdict, dataclass, field


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
    schedule: dict = field(default_factory=dict)  # {tile_m, tile_n, tile_k, warps, stages}
    original_intermediate_memory_mb: float = 0.0
    planned_intermediate_memory_mb: float = 0.0
    generated_kernels: int = 0
    validation_max_error: float = 0.0

    def render(self) -> str:
        """Render the report as a fixed-width human-readable string."""
        d = asdict(self)
        lines = [
            f"Model: {d['model_name']}",
            "",
            f"Captured FX nodes:              {d['captured_fx_nodes']}",
            f"Imported MLIR operations:       {d['imported_mlir_operations']}",
            f"Operations after canonicalize:  {d['operations_after_canonicalize']}",
            f"Fusion groups:                  {d['fusion_groups']}",
            "",
            f"Selected target:                {d['selected_target']}",
            f"Alternative Astra cost:         {d['alternative_astra_cost']}",
            f"Alternative FPGA cost:          {d['alternative_fpga_cost']}",
            "",
            "Selected schedule:",
            f"  tile M:                       {d['schedule'].get('tile_m', '-')}",
            f"  tile N:                       {d['schedule'].get('tile_n', '-')}",
            f"  tile K:                       {d['schedule'].get('tile_k', '-')}",
            f"  warps:                        {d['schedule'].get('warps', '-')}",
            f"  stages:                       {d['schedule'].get('stages', '-')}",
            "",
            f"Original intermediate memory:   {d['original_intermediate_memory_mb']:.1f} MB",
            f"Planned intermediate memory:     {d['planned_intermediate_memory_mb']:.1f} MB",
            f"Generated kernels:               {d['generated_kernels']}",
            f"Validation max error:            {d['validation_max_error']:.4f}",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize all fields to a stable JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> CompilationReport:
        """Inverse of :meth:`to_json`; preserves every dataclass field."""
        return cls(**json.loads(s))
