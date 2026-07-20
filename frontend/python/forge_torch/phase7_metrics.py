"""Phase 7 spec §19 metric table (Task 24).

The spec §19 demo requires a single comparison table that captures the
6 metrics (``latency_us``, ``peak_memory_mb``, ``kernel_count``,
``data_transferred_mb``, ``compile_time_ms``, ``numerical_error_max``)
for each of the three configurations (A / B / C) the heterogeneous
partitioner dispatches.

This module is the canonical source of those 6 fields. The five Phase
7 e2e tests (Tasks 21-23 + the 4-case metrics-table test in Task 24)
all populate the same :class:`Metrics` dataclass; the
:class:`MetricsTable` holds all three configs side-by-side and renders
the spec §19 markdown layout.

Field provenance
----------------

* ``latency_us`` — wall-clock for one ``compiled(*inputs)`` call
  (synchronous execution through the FX Interpreter, Phase 1 stub).
* ``peak_memory_mb`` — max over regions of the region's live-tensor
  byte count; the partitioner is a stub so the live-set equals the
  model's intermediate-tensor size. Computed from the region's input
  shapes (the captured FX graph's per-node types are not exposed on
  ``CompiledModel`` in the shipped scope).
* ``kernel_count`` — ``len(compiled.regions)``; one region per
  per-op FX node after the adjacent-merge step.
* ``data_transferred_mb`` — per-region byte accounting with
  backend-specific transfer multipliers: triton in-process (0x),
  astra DMA (2x input + output bytes per matmul region), microgpu
  LOAD/STORE (2x per matmul region). The multiplier is the spec §19
  "DMA transfers" signal — Config B (matmul → astra) is therefore
  strictly larger than Config A (all → triton, in-process).
* ``compile_time_ms`` — wall-clock for the
  ``compile_partitioned(model, inputs, config=config)`` call.
* ``numerical_error_max`` — ``max(|compiled(*inputs) -
  model(*inputs)|)``; the FX Interpreter delegates to eager so the
  error is 0.0 in the shipped Phase 1 stub path. The field is
  retained for the Phase 7 lowered-execution path that lands once
  ForgeRT Phase 5 is live.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Final

import torch

from forge_torch.compile_partitioned import (
    CompiledModel,
    ValidConfig,
    compile_partitioned,
)

# Backend-specific transfer multipliers (spec §19 "data transferred").
#   * triton — in-process kernel launch; no host<->device transfer.
#   * astra  — Phase 4 microcode DMA: every matmul region issues
#              DMA_LOAD + DMA_STORE, so 2x the tensor byte count.
#   * microgpu — Phase 8 SIMT interpreter: LOAD_GLOBAL + STORE_GLOBAL
#              per matmul region, same 2x accounting as astra.
_TRANSFER_MULTIPLIER: Final[dict[str, float]] = {
    "triton": 0.0,
    "astra": 2.0,
    "microgpu": 2.0,
}

# Regions whose op set is purely ``forge.tensor.gemm`` route to
# astra / microgpu in Config B / C; the other regions stay on triton
# (the most permissive backend). This matches the per-region decision
# in :mod:`forge_torch.compile_partitioned`.
_MATMUL_OPS: Final[frozenset[str]] = frozenset({"forge.tensor.gemm"})


@dataclass(frozen=True, slots=True)
class Metrics:
    """Spec §19 metrics for a single configuration (one of A / B / C).

    All six fields are populated by :func:`metrics_table`. Values are
    non-NaN, non-negative, and expressed in the units named in the
    field (microseconds for latency, megabytes for memory, etc.).
    """

    latency_us: float
    peak_memory_mb: float
    kernel_count: int
    data_transferred_mb: float
    compile_time_ms: float
    numerical_error_max: float

    def __post_init__(self) -> None:
        # Sentinel check: a NaN or negative value indicates the metric
        # was not populated; raise immediately so the caller knows the
        # table is incomplete rather than silently reporting garbage.
        for name, value in (
            ("latency_us", self.latency_us),
            ("peak_memory_mb", self.peak_memory_mb),
            ("data_transferred_mb", self.data_transferred_mb),
            ("compile_time_ms", self.compile_time_ms),
            ("numerical_error_max", self.numerical_error_max),
        ):
            if math.isnan(value):
                msg = f"Metrics.{name} is NaN — measurement did not populate"
                raise ValueError(msg)
            if value < 0.0:
                msg = f"Metrics.{name} is negative ({value}); metrics must be non-negative"
                raise ValueError(msg)
        if self.kernel_count < 0:
            msg = f"Metrics.kernel_count is negative ({self.kernel_count})"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MetricsTable:
    """Spec §19 comparison table — one ``Metrics`` per configuration.

    ``render()`` produces the spec §19 markdown layout: 4 rows
    (``Metric`` header + ``Config A`` / ``Config B`` / ``Config C``)
    and 7 columns (the row-label ``Metric`` + 6 metric-value columns).
    """

    config_a: Metrics
    config_b: Metrics
    config_c: Metrics

    def render(self) -> str:
        """Render the spec §19 comparison table as a markdown string."""
        rows = [
            (
                "latency_us",
                self.config_a.latency_us,
                self.config_b.latency_us,
                self.config_c.latency_us,
            ),
            (
                "peak_memory_mb",
                self.config_a.peak_memory_mb,
                self.config_b.peak_memory_mb,
                self.config_c.peak_memory_mb,
            ),
            (
                "kernel_count",
                self.config_a.kernel_count,
                self.config_b.kernel_count,
                self.config_c.kernel_count,
            ),
            (
                "data_transferred_mb",
                self.config_a.data_transferred_mb,
                self.config_b.data_transferred_mb,
                self.config_c.data_transferred_mb,
            ),
            (
                "compile_time_ms",
                self.config_a.compile_time_ms,
                self.config_b.compile_time_ms,
                self.config_c.compile_time_ms,
            ),
            (
                "numerical_error_max",
                self.config_a.numerical_error_max,
                self.config_b.numerical_error_max,
                self.config_c.numerical_error_max,
            ),
        ]
        # Transposed layout: 4 rows (Metric / Config A / Config B /
        # Config C) x 7 columns (row label + 6 metric values). Each
        # column is a metric; each row is a view (header + 3 configs).
        header = "| Metric   | " + " | ".join(name for name, *_ in rows) + " |"
        separator = "|" + "|".join(["---"] * (len(rows) + 1)) + "|"
        body_lines = [
            "| Config A | " + " | ".join(_format(a) for _, a, _, _ in rows) + " |",
            "| Config B | " + " | ".join(_format(b) for _, _, b, _ in rows) + " |",
            "| Config C | " + " | ".join(_format(c) for _, _, _, c in rows) + " |",
        ]
        return "\n".join([header, separator, *body_lines]) + "\n"


def _format(value: float) -> str:
    """Format a metric value for the markdown table."""
    if isinstance(value, int) or value == int(value):
        return str(int(value))
    return f"{value:.4g}"


# ---------------------------------------------------------------------------
# Measurement helpers.
# ---------------------------------------------------------------------------


def _region_input_bytes(compiled: CompiledModel) -> int:
    """Estimate the byte count of the per-region input tensor.

    The shipped :class:`CompiledModel` does not expose per-region
    tensor shapes; the partitioner is a stub and the FX Interpreter
    delegates to eager. The model's first example input is a good
    proxy for the per-region live-set size — every region reads /
    writes a tensor of roughly the same byte count as the model
    input for the spec §19 transformer block on the brief's
    ``(B=1, S=4, d_model=64)`` shape.
    """
    if not compiled.example_inputs:
        return 0
    sample = compiled.example_inputs[0]
    return sample.numel() * sample.element_size()


def _is_pure_matmul_region(region: list) -> bool:
    """``True`` if every mapped op in ``region`` is a matmul op."""
    from forge_torch.partitioner import map_node_to_forge_op

    op_names: set[str] = set()
    for node in region:
        mapped = map_node_to_forge_op(node)
        if mapped is not None:
            op_names.add(mapped)
    return bool(op_names) and op_names <= _MATMUL_OPS


def _per_config_transfer_mb(
    compiled: CompiledModel,
    per_region_bytes: int,
) -> float:
    """Sum the per-region data-transferred bytes for ``compiled``."""
    total_bytes = 0
    for region, backend in zip(compiled.regions, compiled.region_assignments, strict=True):
        multiplier = _TRANSFER_MULTIPLIER.get(backend, 0.0)
        if multiplier == 0.0:
            continue
        # Only matmul regions incur DMA / LOAD / STORE on the shipped
        # backends; elementwise regions reuse the in-flight tensor
        # without host↔device movement.
        if not _is_pure_matmul_region(region):
            continue
        total_bytes += int(multiplier * per_region_bytes)
    return total_bytes / (1024 * 1024)


def _peak_memory_mb(compiled: CompiledModel, per_region_bytes: int) -> float:
    """Peak live-tensor memory across the partitioner's regions."""
    if not compiled.regions:
        return 0.0
    return per_region_bytes / (1024 * 1024)


def metrics_table(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    config: ValidConfig,
) -> Metrics:
    """Run ``compile_partitioned`` with ``config`` and capture the 6 metrics.

    The returned :class:`Metrics` is the spec §19 row for one
    configuration. Use :func:`build_metrics_table` to get a
    comparison table covering all three configs.
    """
    model.eval()
    with torch.no_grad():
        eager_output = model(*inputs)

    t0 = time.perf_counter()
    compiled: CompiledModel = compile_partitioned(model, inputs, config=config)
    compile_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    with torch.no_grad():
        compiled_output = compiled(*inputs)
    latency_us = (time.perf_counter() - t0) * 1_000_000.0

    numerical_error_max = float((compiled_output - eager_output).abs().max().item())
    per_region_bytes = _region_input_bytes(compiled)
    data_transferred_mb = _per_config_transfer_mb(compiled, per_region_bytes)
    peak_memory_mb = _peak_memory_mb(compiled, per_region_bytes)

    return Metrics(
        latency_us=latency_us,
        peak_memory_mb=peak_memory_mb,
        kernel_count=compiled.num_regions,
        data_transferred_mb=data_transferred_mb,
        compile_time_ms=compile_time_ms,
        numerical_error_max=numerical_error_max,
    )


def build_metrics_table(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
) -> MetricsTable:
    """Run all three configs (A / B / C) and return the comparison table."""
    return MetricsTable(
        config_a=metrics_table(model, inputs, "A"),
        config_b=metrics_table(model, inputs, "B"),
        config_c=metrics_table(model, inputs, "C"),
    )


__all__ = (
    "Metrics",
    "MetricsTable",
    "build_metrics_table",
    "metrics_table",
)
