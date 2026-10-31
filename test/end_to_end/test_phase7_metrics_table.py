"""Phase 7 spec §19 metric table pytest (Task 24, 4 cases).

The spec §19 demo requires a single comparison table that captures
the 6 metrics (``latency_us``, ``peak_memory_mb``, ``kernel_count``,
``data_transferred_mb``, ``compile_time_ms``, ``numerical_error_max``)
for each of the three configurations (A / B / C). This test file
exercises :mod:`forge_torch.phase7_metrics` end-to-end:

  1. ``test_metrics_table_populated`` — every field is populated
     (non-NaN, non-negative) for each of the three configs.
  2. ``test_metrics_table_compile_time`` — ``compile_time_ms`` is
     strictly positive for every config.
  3. ``test_metrics_table_data_transferred_differs_between_configs`` —
     Config B's ``data_transferred_mb`` exceeds Config A's because
     Config B routes matmul regions to Astra-sim, which uses DMA
     transfers.
  4. ``test_metrics_table_render_matches_spec_layout`` — the
     ``render()`` output is a markdown table with 4 rows (Metric /
     Config A / Config B / Config C) and the 6 metric value columns.

The function-style TransformerBlock mirror mirrors the spec §19
topology using only ops the shipped FX importer lowers today — see
``test_phase7_config_a.py`` for the rationale.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge_torch.phase7_metrics import (
    MetricsTable,
    build_metrics_table,
)

# ---------------------------------------------------------------------------
# Function-style TransformerBlock mirror — duplicated from
# ``test_phase7_config_a.py`` so each test file is self-contained.
# ---------------------------------------------------------------------------


class _Spec19Transformer(nn.Module):
    """Spec §19 topology in FX-traceable form (matmul/add/gelu only)."""

    def forward(
        self,
        x: torch.Tensor,  # (B, S, d_model)
        w_qkv: torch.Tensor,
        b_qkv: torch.Tensor,
        w_attn: torch.Tensor,
        b_attn: torch.Tensor,
        w_out: torch.Tensor,
        b_out: torch.Tensor,
        w_mlp1: torch.Tensor,
        b_mlp1: torch.Tensor,
        w_mlp2: torch.Tensor,
        b_mlp2: torch.Tensor,
        w_ln1: torch.Tensor,
        b_ln1: torch.Tensor,
        w_ln2: torch.Tensor,
        b_ln2: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.matmul(x, w_ln1) + b_ln1  # LN1 approximation
        qkv = torch.matmul(h, w_qkv) + b_qkv
        attn = torch.matmul(qkv, w_attn) + b_attn
        x = x + torch.matmul(attn, w_out) + b_out
        h = torch.matmul(x, w_ln2) + b_ln2  # LN2 approximation
        h = torch.matmul(h, w_mlp1) + b_mlp1
        h = F.gelu(h)
        h = torch.matmul(h, w_mlp2) + b_mlp2
        x = x + h
        return x


D_MODEL = 64
N_HEADS = 4
D_FF = 128


def _make_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    B, S, D = 1, 4, D_MODEL
    d_ff = D_FF
    return [
        torch.randn(B, S, D, dtype=torch.float32),
        torch.randn(D, 3 * D, dtype=torch.float32),
        torch.randn(3 * D, dtype=torch.float32),
        torch.randn(3 * D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, d_ff, dtype=torch.float32),
        torch.randn(d_ff, dtype=torch.float32),
        torch.randn(d_ff, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
    ]


@pytest.fixture
def spec19_model() -> _Spec19Transformer:
    return _Spec19Transformer()


@pytest.fixture
def spec19_inputs() -> list[torch.Tensor]:
    return _make_inputs()


@pytest.fixture
def table(
    spec19_model: _Spec19Transformer,
    spec19_inputs: list[torch.Tensor],
) -> MetricsTable:
    return build_metrics_table(spec19_model, spec19_inputs)


# ---------------------------------------------------------------------------
# Case 1 — all 6 fields are populated (non-NaN, non-negative) for each
# config.
# ---------------------------------------------------------------------------


def test_metrics_table_populated(table: MetricsTable) -> None:
    """Every spec §19 field is populated for each of the 3 configs."""
    for config_name, metrics in (
        ("config_a", table.config_a),
        ("config_b", table.config_b),
        ("config_c", table.config_c),
    ):
        for field in (
            "latency_us",
            "peak_memory_mb",
            "data_transferred_mb",
            "compile_time_ms",
            "numerical_error_max",
        ):
            value = getattr(metrics, field)
            assert not math.isnan(value), f"{config_name}.{field} is NaN"
            assert value >= 0.0, f"{config_name}.{field} is negative ({value})"
        assert metrics.kernel_count > 0, (
            f"{config_name}.kernel_count must be > 0 (got {metrics.kernel_count})"
        )


# ---------------------------------------------------------------------------
# Case 2 — compile_time_ms > 0 for each config.
# ---------------------------------------------------------------------------


def test_metrics_table_compile_time(table: MetricsTable) -> None:
    """``compile_time_ms`` is strictly positive for every config."""
    for config_name, metrics in (
        ("config_a", table.config_a),
        ("config_b", table.config_b),
        ("config_c", table.config_c),
    ):
        assert metrics.compile_time_ms > 0.0, (
            f"{config_name}.compile_time_ms must be > 0 (got {metrics.compile_time_ms})"
        )


# ---------------------------------------------------------------------------
# Case 3 — data_transferred_mb differs between configs.
# Config A (all → triton) is in-process; Config B (matmul → astra)
# uses DMA transfers, so its data_transferred_mb is strictly larger.
# ---------------------------------------------------------------------------


def test_metrics_table_data_transferred_differs_between_configs(
    table: MetricsTable,
) -> None:
    """Config B's data_transferred_mb exceeds Config A's (DMA vs in-process)."""
    assert table.config_b.data_transferred_mb > table.config_a.data_transferred_mb, (
        "Config B must transfer more bytes than Config A "
        "(matmul regions route to astra which uses DMA). "
        f"Got config_a={table.config_a.data_transferred_mb:.6e} MB, "
        f"config_b={table.config_b.data_transferred_mb:.6e} MB"
    )


# ---------------------------------------------------------------------------
# Case 4 — render() matches the spec §19 layout: 4 rows (Metric header
# + Config A / B / C) and 6 metric value columns.
# ---------------------------------------------------------------------------


def test_metrics_table_render_matches_spec_layout(table: MetricsTable) -> None:
    """``render()`` produces the spec §19 markdown comparison table."""
    rendered = table.render()

    # Markdown rows: lines that start with ``|`` and contain at least
    # one non-separator, non-pipe-only cell. The standard markdown
    # separator row (``|---|---|...``) is excluded.
    data_rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("|") and not set(line.replace("|", "").strip()) <= {"-"}
    ]
    assert len(data_rows) == 4, (
        f"render() must produce 4 data rows (Metric + Config A/B/C); "
        f"got {len(data_rows)}:\n{rendered}"
    )

    # Row 1: header naming the 6 metric columns.
    header = data_rows[0]
    for metric_name in (
        "latency_us",
        "peak_memory_mb",
        "kernel_count",
        "data_transferred_mb",
        "compile_time_ms",
        "numerical_error_max",
    ):
        assert metric_name in header, (
            f"render() header must name the {metric_name!r} column; got:\n{header}"
        )

    # Rows 2-4: one per config, each starting with the config name.
    assert "Config A" in data_rows[1]
    assert "Config B" in data_rows[2]
    assert "Config C" in data_rows[3]

    # The header row must declare exactly 7 columns (Metric label + 6
    # metric value columns) — the 6 metric columns satisfy the
    # spec §19 layout requirement.
    header_columns = [cell.strip() for cell in header.strip("|").split("|")]
    assert len(header_columns) == 7, (
        f"render() header must have 7 columns (Metric + 6 metric values); "
        f"got {len(header_columns)}: {header_columns}"
    )

    # Each config row must have 7 cells matching the header columns.
    for row in data_rows[1:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 7, f"render() config row must have 7 cells; got {len(cells)}: {cells}"
