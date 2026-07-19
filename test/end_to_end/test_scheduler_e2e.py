"""Scheduler and MemoryPlanner end-to-end pytest (Task 34)."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as element_tree
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
TEST_SCHEDULER = REPO / "build" / "runtime" / "tests" / "test_scheduler"


def test_scheduler_uses_planned_offsets_in_mock_provider_log(
    fused_mlp: torch.nn.Module,
    tmp_path: Path,
) -> None:
    assert fused_mlp.__class__.__name__ == "FusedMLP"
    assert TEST_SCHEDULER.is_file(), f"scheduler test binary is missing: {TEST_SCHEDULER}"
    result_path = tmp_path / "scheduler-results.xml"

    completed = subprocess.run(
        [
            str(TEST_SCHEDULER),
            "--gtest_filter=Scheduler.UsesMemoryPlannerAllocations",
            f"--gtest_output=xml:{result_path}",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"Scheduler gtest failed with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    root = element_tree.parse(result_path).getroot()
    offset_property = root.find(".//property[@name='offset_log']")
    assert offset_property is not None
    assert offset_property.attrib["value"] == "128,64,192->0;0->64;64->0"
