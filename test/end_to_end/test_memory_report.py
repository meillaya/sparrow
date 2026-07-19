"""End-to-end correctness check for compilation-report memory metrics (Task 38)."""

from __future__ import annotations

from pathlib import Path
from typing import Final
from unittest.mock import patch

import torch

from frontend.python.forge_torch.backend import compile
from frontend.python.forge_torch.cache_store import CacheStore
from frontend.python.forge_torch.report import CompilationReport

MEBIBYTE: Final = 1024 * 1024


def test_compiled_fused_mlp_report_planned_memory_is_output_bounded(
    fused_mlp: torch.nn.Module,
    tmp_path: Path,
) -> None:
    x = torch.randn(1, 1024, 32, dtype=torch.float32)
    weight = torch.randn(32, 2048, dtype=torch.float32)
    bias = torch.randn(2048, dtype=torch.float32)
    captured: list[CompilationReport] = []

    with patch(
        "frontend.python.forge_torch.backend._emit_report",
        side_effect=captured.append,
    ):
        compiled = compile(
            fused_mlp,
            [x, weight, bias],
            cache_store=CacheStore(tmp_path / "cache"),
        )

    assert compiled.compiled
    assert len(captured) == 1
    report = captured[0]
    output_buffer_mb = (x.shape[0] * x.shape[1] * weight.shape[1] * 4) / MEBIBYTE
    assert report.planned_intermediate_memory_mb > 0.0
    assert report.planned_intermediate_memory_mb < output_buffer_mb * 1.5
