"""Compilation-report memory fields.

``compile()`` does not build an execution plan, so it must not invent
memory numbers. The report's plan-driven path is tested directly with a
small allocation view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from frontend.python.forge_torch.backend import compile
from frontend.python.forge_torch.cache_store import CacheStore
from frontend.python.forge_torch.report import CompilationReport


@dataclass(frozen=True)
class _Allocation:
    byte_offset: int
    byte_size: int


@dataclass(frozen=True)
class _Plan:
    allocations: tuple[_Allocation, ...]


def test_compiled_report_leaves_memory_unavailable(
    fused_mlp: torch.nn.Module,
    tmp_path: Path,
) -> None:
    """A compile-only report does not claim a memory plan."""
    x = torch.randn(1, 1024, 32, dtype=torch.float32)
    weight = torch.randn(32, 2048, dtype=torch.float32)
    bias = torch.randn(2048, dtype=torch.float32)

    compiled = compile(
        fused_mlp,
        [x, weight, bias],
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert compiled.compiled
    assert compiled.report is not None
    assert compiled.report.planned_intermediate_memory_mb is None
    assert compiled.report.original_intermediate_memory_mb is None


def test_report_computes_memory_from_a_plan() -> None:
    """When a plan is supplied, the report derives both memory fields."""
    mebibyte = 1024 * 1024
    plan = _Plan(
        allocations=(
            _Allocation(byte_offset=0, byte_size=mebibyte),
            _Allocation(byte_offset=mebibyte, byte_size=2 * mebibyte),
        )
    )
    report = CompilationReport(model_name="FusedMLP", plan=plan)

    assert report.original_intermediate_memory_mb == 3.0
    assert report.planned_intermediate_memory_mb == 3.0
