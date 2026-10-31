"""Phase 7 spec §19 metric table (Task 24).

The spec §19 demo requires a single comparison table that captures the
6 metrics (``latency_us``, ``peak_memory_mb``, ``kernel_count``,
``data_transferred_mb``, ``compile_time_ms``, ``numerical_error_max``)
for each of the three configurations (A / B / C) the heterogeneous
partitioner dispatches. Config A executes a real CPU plan; configs B
and C are absent because their lowerings do not exist. Memory values
remain analytical estimates, and ``MetricsTable.render`` marks them.

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
* ``data_transferred_mb`` — bytes moved across a device boundary. The
  only runnable configuration is CPU-only, so this is 0.0. Synthetic
  transfer multipliers are gone; a backend that actually crosses a
  device boundary must report its measured bytes.
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
from dataclasses import dataclass, field
from typing import Final

import torch

from forge_torch.compile_partitioned import (
    CompiledModel,
    ValidConfig,
    compile_partitioned,
)

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
    #: True while memory and transfer values are analytical estimates.
    estimated: bool = True

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
    config_b: Metrics | None = None
    config_c: Metrics | None = None
    notes: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """Render the spec §19 comparison table as a markdown string.

        Configs whose backend lowering is unavailable render ``n/a`` and
        the reason is appended as a footnote. Config A is always present.
        """
        rows = [
            ("latency_us", "latency_us"),
            ("peak_memory_mb", "peak_memory_mb"),
            ("kernel_count", "kernel_count"),
            ("data_transferred_mb", "data_transferred_mb"),
            ("compile_time_ms", "compile_time_ms"),
            ("numerical_error_max", "numerical_error_max"),
        ]
        header = "| Metric   | " + " | ".join(name for name, _ in rows) + " |"
        separator = "|" + "|".join(["---"] * (len(rows) + 1)) + "|"
        lines = [header, separator]
        for label, metrics in (
            ("Config A", self.config_a),
            ("Config B", self.config_b),
            ("Config C", self.config_c),
        ):
            cells = [
                _format(getattr(metrics, field)) if metrics is not None else "n/a"
                for _, field in rows
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "Estimated fields: peak_memory_mb, data_transferred_mb. "
            "Config A executes the CPU plan; B and C are not lowered."
        )
        for config, reason in sorted(self.notes.items()):
            lines.append(f"{config}: {reason}")
        return "\n".join(lines) + "\n"


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
    # The CPU plan runs in-process; nothing crosses a device boundary, so
    # the measured transfer count is zero. Backends that do move bytes
    # must report their own measurement rather than a multiplier.
    data_transferred_mb = 0.0
    peak_memory_mb = _peak_memory_mb(compiled, per_region_bytes)

    return Metrics(
        latency_us=latency_us,
        peak_memory_mb=peak_memory_mb,
        kernel_count=compiled.num_regions,
        data_transferred_mb=data_transferred_mb,
        compile_time_ms=compile_time_ms,
        numerical_error_max=numerical_error_max,
        estimated=True,
    )


def build_metrics_table(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
) -> MetricsTable:
    """Run every available config and record why the others are missing."""
    rows: dict[str, Metrics] = {}
    notes: dict[str, str] = {}
    for config in ("A", "B", "C"):
        try:
            rows[config] = metrics_table(model, inputs, config)
        except NotImplementedError as exc:
            notes[config] = str(exc)
    return MetricsTable(
        config_a=rows["A"],
        config_b=rows.get("B"),
        config_c=rows.get("C"),
        notes=notes,
    )


__all__ = (
    "Metrics",
    "MetricsTable",
    "build_metrics_table",
    "metrics_table",
)
