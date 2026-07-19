"""ForgeStack compilation report pytest tests (Task 14, spec §16).

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

from forge_torch.report import CompilationReport

# Section headings the spec §16 example requires. Each appears verbatim in
# the rendered report; the assertions below guard against accidental
# truncation when someone edits the render() template.
REQUIRED_SECTIONS = [
    "Model:",
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


def _populated_report() -> CompilationReport:
    return CompilationReport(
        model_name="FusedMLP",
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
        original_intermediate_memory_mb=45.0,
        planned_intermediate_memory_mb=8.0,
        generated_kernels=1,
        validation_max_error=0.0,
    )


def test_report_render_contains_required_sections() -> None:
    """All 18 spec §16 section headings appear in the rendered output."""
    rendered = _populated_report().render()
    missing = [section for section in REQUIRED_SECTIONS if section not in rendered]
    assert not missing, f"rendered report missing required sections: {missing}"
    # The model name and the FX/MLIR counts are echoed verbatim.
    assert "FusedMLP" in rendered
    assert "triton" in rendered


def test_report_roundtrip_json() -> None:
    """``to_json`` → ``from_json`` preserves every dataclass field."""
    original = _populated_report()
    restored = CompilationReport.from_json(original.to_json())

    for field_name in (
        "model_name",
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

    # Numeric defaults are 0 / 0.0 and render as 0.0 / 0.
    assert "Captured FX nodes:              0" in rendered
    assert "Fusion groups:                  0" in rendered
    assert "Generated kernels:               0" in rendered
    assert "Original intermediate memory:   0.0 MB" in rendered


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
