"""Compilation-report-completeness pytest (Phase 2, Task 22, spec §16).

Runs ``compile(FusedMLP(), inputs)`` through the full Phase-1 pipeline,
captures the emitted :class:`~forge_torch.report.CompilationReport`
(via :mod:`unittest.mock.patch` on
:func:`forge_torch.backend._emit_report`), and asserts every spec §16
section is present and non-empty:

  - model name (sanity)
  - FX node count (``captured_fx_nodes``)
  - MLIR op count per pass (``imported_mlir_operations`` + ``operations_after_canonicalize``)
  - fusion groups (``fusion_groups``)
  - selected target (``selected_target``)
  - alternative target costs (``alternative_astra_cost``, ``alternative_fpga_cost``)
  - schedule (tile_m/n/k, warps, stages)
  - peak intermediate memory (``original_intermediate_memory_mb``, ``planned_intermediate_memory_mb``)
  - generated kernels (``generated_kernels``)
  - validation max error (``validation_max_error``)

Patches the private emitter rather than reading the rendered text back
from disk, so the assertion uses the structured dataclass and is not
brittle to whitespace changes in ``render()``.
"""

from __future__ import annotations

import torch
from forge_torch.conftest import FusedMLP
from forge_torch.report import CompilationReport


def _capture_report_via_patch(tmp_path) -> tuple[CompilationReport, object]:
    """Run compile() and capture the :class:`CompilationReport` it emits.

    Returns ``(report, compiled)`` so callers can both assert on the
    report fields and rerun the compiled callable for end-to-end
    coverage. Uses ``unittest.mock.patch`` on
    :func:`forge_torch.backend._emit_report` because the Phase-1 API
    surface doesn't expose the report object directly.

    The compile() call uses a per-test tmp_path cache root and
    cache_store so the cold-path branch (which is the only path that
    emits a report) fires reliably — the default ``.cache/forgestack``
    is shared across tests and once warm, _emit_report is skipped.
    """
    from pathlib import Path
    from unittest.mock import patch

    from forge_torch.backend import compile
    from forge_torch.cache_store import CacheStore

    captured: list[CompilationReport] = []

    def _capture(report: CompilationReport) -> None:
        captured.append(report)

    cache_root: Path = tmp_path / "cache"
    store = CacheStore(cache_root)

    with patch("forge_torch.backend._emit_report", side_effect=_capture):
        model = FusedMLP()
        torch.manual_seed(0)
        x = torch.randn(2, 16, 128, dtype=torch.float32)
        w = torch.randn(128, 256, dtype=torch.float32)
        b = torch.randn(256, dtype=torch.float32)
        compiled = compile(model, [x, w, b], cache_store=store)

    assert len(captured) == 1, f"expected 1 emitted report, got {len(captured)}"
    return captured[0], compiled


def test_report_has_model_name(tmp_path) -> None:
    report, _ = _capture_report_via_patch(tmp_path)
    assert report.model_name == "FusedMLP", (
        f"expected model_name='FusedMLP', got {report.model_name!r}"
    )


def test_report_has_fx_node_count(tmp_path) -> None:
    """spec §16: FX node count must be populated (non-zero for a non-trivial graph)."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert isinstance(report.captured_fx_nodes, int)
    assert report.captured_fx_nodes > 0, (
        f"captured_fx_nodes must be > 0 for FusedMLP, got {report.captured_fx_nodes}"
    )


def test_report_has_mlir_op_count_per_pass(tmp_path) -> None:
    """spec §16: MLIR op count, both pre- and post-canonicalize, must be populated."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert isinstance(report.imported_mlir_operations, int)
    assert report.imported_mlir_operations > 0, (
        f"imported_mlir_operations must be > 0, got {report.imported_mlir_operations}"
    )
    assert isinstance(report.operations_after_canonicalize, int)
    assert report.operations_after_canonicalize > 0, (
        f"operations_after_canonicalize must be > 0, got {report.operations_after_canonicalize}"
    )


def test_report_has_fusion_groups(tmp_path) -> None:
    """spec §16: fusion_groups must be populated (≥ 1 for the FusedMLP triple)."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert isinstance(report.fusion_groups, int)
    assert report.fusion_groups >= 1, (
        f"fusion_groups must be ≥ 1 for FusedMLP, got {report.fusion_groups}"
    )


def test_report_has_selected_target(tmp_path) -> None:
    """spec §16: selected_target must be a non-empty target name (triton by default)."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert isinstance(report.selected_target, str)
    assert len(report.selected_target) > 0, "selected_target must be non-empty"
    assert report.selected_target == "triton", (
        f"expected selected_target='triton', got {report.selected_target!r}"
    )


def test_report_has_alternative_target_costs(tmp_path) -> None:
    """spec §16: alternative costs (Astra + FPGA) must be non-empty cost strings."""
    report, _ = _capture_report_via_patch(tmp_path)
    for field_name in ("alternative_astra_cost", "alternative_fpga_cost"):
        value = getattr(report, field_name)
        assert isinstance(value, str)
        assert len(value) > 0, f"{field_name} must be non-empty, got {value!r}"


def test_report_schedule_has_all_tile_knobs(tmp_path) -> None:
    """spec §16: schedule dict must carry tile_m/n/k + warps + stages, all positive ints."""
    report, _ = _capture_report_via_patch(tmp_path)
    schedule = report.schedule
    assert isinstance(schedule, dict), f"schedule must be a dict, got {type(schedule)}"
    for key in ("tile_m", "tile_n", "tile_k", "warps", "stages"):
        assert key in schedule, f"schedule missing required key: {key}"
        value = schedule[key]
        assert isinstance(value, int), f"schedule[{key!r}] must be int, got {type(value).__name__}"
        assert value > 0, f"schedule[{key!r}] must be positive, got {value}"
    assert schedule["tile_m"] == 64
    assert schedule["tile_n"] == 128
    assert schedule["tile_k"] == 32
    assert schedule["warps"] == 8
    assert schedule["stages"] == 4


def test_report_has_peak_intermediate_memory(tmp_path) -> None:
    """spec §16: both original + planned intermediate memory must be > 0."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert report.original_intermediate_memory_mb > 0.0, (
        f"original_intermediate_memory_mb must be > 0, got {report.original_intermediate_memory_mb}"
    )
    assert report.planned_intermediate_memory_mb > 0.0, (
        f"planned_intermediate_memory_mb must be > 0, got {report.planned_intermediate_memory_mb}"
    )
    assert report.planned_intermediate_memory_mb <= report.original_intermediate_memory_mb, (
        f"planned ({report.planned_intermediate_memory_mb}) must be <= "
        f"original ({report.original_intermediate_memory_mb})"
    )


def test_report_has_generated_kernels(tmp_path) -> None:
    """spec §16: generated_kernels must be ≥ 1 (the single fused_gemm_bias_gelu kernel)."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert isinstance(report.generated_kernels, int)
    assert report.generated_kernels >= 1, (
        f"generated_kernels must be ≥ 1, got {report.generated_kernels}"
    )


def test_report_has_validation_max_error(tmp_path) -> None:
    """spec §16: validation_max_error must be present (a float, populated by validate())."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert isinstance(report.validation_max_error, float)
    assert report.validation_max_error >= 0.0, (
        f"validation_max_error must be ≥ 0, got {report.validation_max_error}"
    )


def test_report_full_completeness_single_check(tmp_path) -> None:
    """Composite: every spec §16 field is present + non-empty in one assertion.

    Reads each dataclass field via :func:`getattr` and asserts the
    value is not None and (for strings) non-empty. Acts as the spec
    §16 smoke gate that all the per-field tests above also pin
    individually.
    """
    report, compiled = _capture_report_via_patch(tmp_path)

    required_strings = {
        "model_name": "FusedMLP",
        "selected_target": "triton",
        "alternative_astra_cost": None,
        "alternative_fpga_cost": None,
    }
    for field_name, expected in required_strings.items():
        value = getattr(report, field_name)
        assert value is not None, f"{field_name} must not be None"
        assert isinstance(value, str), f"{field_name} must be str, got {type(value).__name__}"
        assert len(value) > 0, f"{field_name} must be non-empty"
        if expected is not None:
            assert value == expected, f"{field_name} expected {expected!r}, got {value!r}"

    required_ints = {
        "captured_fx_nodes": (1, None),
        "imported_mlir_operations": (1, None),
        "operations_after_canonicalize": (1, None),
        "fusion_groups": (1, None),
        "generated_kernels": (1, None),
    }
    for field_name, (lo, hi) in required_ints.items():
        value = getattr(report, field_name)
        assert isinstance(value, int), f"{field_name} must be int, got {type(value).__name__}"
        assert value >= lo, f"{field_name} must be ≥ {lo}, got {value}"
        if hi is not None:
            assert value <= hi, f"{field_name} must be ≤ {hi}, got {value}"

    required_floats = {
        "original_intermediate_memory_mb": (0.0, None),
        "planned_intermediate_memory_mb": (0.0, None),
        "validation_max_error": (0.0, None),
    }
    for field_name, (lo, hi) in required_floats.items():
        value = getattr(report, field_name)
        assert isinstance(value, float), f"{field_name} must be float, got {type(value).__name__}"
        assert value >= lo, f"{field_name} must be ≥ {lo}, got {value}"
        if hi is not None:
            assert value <= hi, f"{field_name} must be ≤ {hi}, got {value}"

    schedule = report.schedule
    for key in ("tile_m", "tile_n", "tile_k", "warps", "stages"):
        assert key in schedule, f"schedule missing required key: {key}"
        v = schedule[key]
        assert isinstance(v, int) and v > 0, f"schedule[{key!r}] must be positive int, got {v!r}"

    assert compiled.mlir_text  # The compiled call produced MLIR text.
    assert "forge.graph" in compiled.mlir_text
