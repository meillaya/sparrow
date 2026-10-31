"""Sparrow compilation report pytest tests (Task 14, spec §16).

Four tests pin the :class:`forge_torch.report.CompilationReport` contract:

1. ``test_report_render_contains_required_sections`` — every spec §16
   section heading is rendered to text.
2. ``test_report_roundtrip_json`` — ``to_json`` / ``from_json`` preserves
   every field.
3. ``test_report_default_values`` — an empty report renders without
   raising and contains the expected placeholders.
4. ``test_report_validation_max_error_format`` — the
   ``validation_max_error`` line uses the documented ``{val:.4f}`` format.
"""

from __future__ import annotations

from dataclasses import dataclass

from frontend.python.forge_torch.report import CompilationReport

# Section headings the spec §16 example requires. Each appears verbatim in
# the rendered report; the assertions below guard against accidental
# truncation when someone edits the render() template.
REQUIRED_SECTIONS = [
    "Model:",
    "Execution path:",
    "Captured FX nodes:",
    "Imported MLIR operations:",
    "Operations after canonicalize:",
    "Fusion groups:",
    "Selected target:",
    "Alternative Astra cost:",
    "Alternative FPGA cost:",
    "Selected schedule:",
    "tile M:",
    "tile N:",
    "tile K:",
    "warps:",
    "stages:",
    "Original intermediate memory:",
    "Planned intermediate memory:",
    "Generated kernels:",
    "Validation max error:",
]


@dataclass(frozen=True, slots=True)
class _Allocation:
    byte_offset: int
    byte_size: int


@dataclass(frozen=True, slots=True)
class _ExecutionPlan:
    allocations: tuple[_Allocation, ...]


def _populated_report() -> CompilationReport:
    mebibyte = 1024 * 1024
    plan = _ExecutionPlan(
        allocations=(
            _Allocation(byte_offset=0, byte_size=mebibyte),
            _Allocation(byte_offset=mebibyte, byte_size=2 * mebibyte),
            _Allocation(byte_offset=0, byte_size=mebibyte),
        )
    )
    return CompilationReport(
        model_name="FusedMLP",
        execution_path="eager",
        captured_fx_nodes=3,
        imported_mlir_operations=3,
        operations_after_canonicalize=3,
        fusion_groups=1,
        selected_target="triton",
        alternative_astra_cost="1.31x",
        alternative_fpga_cost="2.04x including transfer",
        schedule={
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 32,
            "warps": 8,
            "stages": 4,
        },
        generated_kernels=1,
        validation_max_error=0.0,
        plan=plan,
    )


def test_report_render_contains_required_sections() -> None:
    """All section headings, including the execution path, appear in the rendered output."""
    report = _populated_report()
    rendered = report.render()
    missing = [section for section in REQUIRED_SECTIONS if section not in rendered]
    assert not missing, f"rendered report missing required sections: {missing}"
    assert report.original_intermediate_memory_mb == 4.0
    assert report.planned_intermediate_memory_mb == 3.0
    # The model name and the FX/MLIR counts are echoed verbatim.
    assert "FusedMLP" in rendered
    assert "triton" in rendered


def test_report_roundtrip_json() -> None:
    """``to_json`` → ``from_json`` preserves every dataclass field."""
    original = _populated_report()
    restored = CompilationReport.from_json(original.to_json())

    for field_name in (
        "model_name",
        "execution_path",
        "captured_fx_nodes",
        "imported_mlir_operations",
        "operations_after_canonicalize",
        "fusion_groups",
        "selected_target",
        "alternative_astra_cost",
        "alternative_fpga_cost",
        "schedule",
        "original_intermediate_memory_mb",
        "planned_intermediate_memory_mb",
        "generated_kernels",
        "validation_max_error",
    ):
        assert getattr(restored, field_name) == getattr(original, field_name), field_name


def test_report_default_values() -> None:
    """An empty :class:`CompilationReport` renders without raising and uses
    the documented placeholders for unset fields.
    """
    report = CompilationReport(model_name="<unset>")
    rendered = report.render()

    # Empty schedule falls back to '-' placeholders for each tile knob.
    assert "tile M:                       -" in rendered
    assert "tile N:                       -" in rendered
    assert "tile K:                       -" in rendered
    assert "warps:                        -" in rendered
    assert "stages:                       -" in rendered

    # Measured counts default to 0. Fields that were never measured render
    # as unavailable rather than as fabricated numbers.
    assert "Captured FX nodes:              0" in rendered
    assert "Fusion groups:                  unavailable" in rendered
    assert "Generated kernels:               unavailable" in rendered
    assert "Original intermediate memory:   unavailable" in rendered
    assert "Validation max error:            unavailable" in rendered


def test_report_validation_max_error_format() -> None:
    """``validation_max_error`` uses the documented ``{val:.4f}`` format."""
    report = _populated_report()
    report.validation_max_error = 0.0017
    rendered = report.render()

    # Four-digit fractional format ⇒ 0.0017.
    assert "Validation max error:            0.0017" in rendered

    # Larger values also format with exactly four fractional digits.
    report.validation_max_error = 12.5
    assert "Validation max error:            12.5000" in report.render()
