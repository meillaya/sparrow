"""Spec §19 metric table: config A measured, B/C fail closed (M4)."""

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


class _Spec19Transformer(nn.Module):
    """Spec §19 topology in FX-traceable form (matmul/add/gelu only)."""

    def forward(
        self,
        x: torch.Tensor,
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
        h = torch.matmul(x, w_ln1) + b_ln1
        qkv = torch.matmul(h, w_qkv) + b_qkv
        attn = torch.matmul(qkv, w_attn) + b_attn
        x = x + torch.matmul(attn, w_out) + b_out
        h = torch.matmul(x, w_ln2) + b_ln2
        h = torch.matmul(h, w_mlp1) + b_mlp1
        h = F.gelu(h)
        h = torch.matmul(h, w_mlp2) + b_mlp2
        return x + h


def _make_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    B, S, D = 1, 4, 64
    d_ff = 128
    return [
        torch.randn(B, S, D),
        torch.randn(D, 3 * D),
        torch.randn(3 * D),
        torch.randn(3 * D, D),
        torch.randn(D),
        torch.randn(D, D),
        torch.randn(D),
        torch.randn(D, d_ff),
        torch.randn(d_ff),
        torch.randn(d_ff, D),
        torch.randn(D),
        torch.randn(D, D),
        torch.randn(D),
        torch.randn(D, D),
        torch.randn(D),
    ]


@pytest.fixture
def table() -> MetricsTable:
    return build_metrics_table(_Spec19Transformer(), _make_inputs())


def test_config_a_metrics_populated(table: MetricsTable) -> None:
    metrics = table.config_a
    for field in (
        "latency_us",
        "peak_memory_mb",
        "data_transferred_mb",
        "compile_time_ms",
        "numerical_error_max",
    ):
        value = getattr(metrics, field)
        assert not math.isnan(value)
        assert value >= 0.0
    assert metrics.kernel_count > 0


def test_config_a_compile_time_is_positive(table: MetricsTable) -> None:
    assert table.config_a.compile_time_ms > 0.0


def test_unlowered_configs_are_absent_and_zero_transfer(table: MetricsTable) -> None:
    """No synthetic transfer multipliers: the CPU plan moves zero device bytes."""
    assert table.config_b is None
    assert table.config_c is None
    assert set(table.notes) == {"B", "C"}
    assert "Astra" in table.notes["B"]
    assert "MicroGPU" in table.notes["C"]
    assert table.config_a.data_transferred_mb == 0.0


def test_render_keeps_spec_layout_and_marks_unavailable(table: MetricsTable) -> None:
    rendered = table.render()
    data_rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("|") and not set(line.replace("|", "").strip()) <= {"-"}
    ]
    assert len(data_rows) == 4
    assert "latency_us" in data_rows[0]
    assert "n/a" in rendered
    assert "B:" in rendered and "C:" in rendered
