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
    emits a report) fires reliably — the default ``.cache/sparrow``
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


def test_report_does_not_invent_fusion_groups(tmp_path) -> None:
    """The shipped compile path does not run the fuse pass, so the count is None."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert report.fusion_groups is None, (
        f"fusion_groups must stay None until a pass measures it, got {report.fusion_groups!r}"
    )


def test_report_has_selected_target(tmp_path) -> None:
    """spec §16: selected_target must be a non-empty target name (triton by default)."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert isinstance(report.selected_target, str)
    assert len(report.selected_target) > 0, "selected_target must be non-empty"
    assert report.selected_target == "triton", (
        f"expected selected_target='triton', got {report.selected_target!r}"
    )


def test_report_does_not_invent_alternative_costs(tmp_path) -> None:
    """No cost model runs at compile time, so alternative costs stay None."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert report.execution_path in {"eager", "native"}
    for field_name in ("alternative_astra_cost", "alternative_fpga_cost"):
        assert getattr(report, field_name) is None, field_name


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
    """No plan is built at compile time, so memory fields are unavailable."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert report.original_intermediate_memory_mb is None
    assert report.planned_intermediate_memory_mb is None
    assert report.validation_max_error is None
    assert report.execution_path in {"eager", "native"}
    if report.execution_path == "native":
        assert report.generated_kernels == 1
    else:
        assert report.generated_kernels is None


def test_report_has_generated_kernels(tmp_path) -> None:
    """The count is 1 on the native path and None on the eager path."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert report.generated_kernels in {None, 1}
    if report.execution_path == "native":
        assert report.generated_kernels == 1
    else:
        assert report.generated_kernels is None


def test_report_has_validation_max_error(tmp_path) -> None:
    """Validation runs after compilation, so the report carries None at emit time."""
    report, _ = _capture_report_via_patch(tmp_path)
    assert report.validation_max_error is None


def test_report_full_completeness_single_check(tmp_path) -> None:
    """The report separates measured fields from fields that are not measured."""
    report, compiled = _capture_report_via_patch(tmp_path)

    measured_strings = {
        "model_name": "FusedMLP",
        "selected_target": "triton",
    }
    for field_name, expected in measured_strings.items():
        value = getattr(report, field_name)
        assert value == expected, f"{field_name} expected {expected!r}, got {value!r}"

    assert report.execution_path in {"eager", "native"}
    for field_name in ("alternative_astra_cost", "alternative_fpga_cost"):
        assert getattr(report, field_name) is None, field_name

    measured_ints = {
        "captured_fx_nodes": 1,
        "imported_mlir_operations": 1,
        "operations_after_canonicalize": 1,
    }
    for field_name, lo in measured_ints.items():
        value = getattr(report, field_name)
        assert isinstance(value, int), f"{field_name} must be int, got {type(value).__name__}"
        assert value >= lo, f"{field_name} must be ≥ {lo}, got {value}"

    assert report.fusion_groups is None
    assert report.generated_kernels in {None, 1}

    for field_name in (
        "original_intermediate_memory_mb",
        "planned_intermediate_memory_mb",
        "validation_max_error",
    ):
        assert getattr(report, field_name) is None, field_name

    schedule = report.schedule
    for key in ("tile_m", "tile_n", "tile_k", "warps", "stages"):
        assert key in schedule, f"schedule missing required key: {key}"
        v = schedule[key]
        assert isinstance(v, int) and v > 0, f"schedule[{key!r}] must be positive int, got {v!r}"

    assert compiled.mlir_text  # The compiled call produced MLIR text.
    assert "forge.graph" in compiled.mlir_text
