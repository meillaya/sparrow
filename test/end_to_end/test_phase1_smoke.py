"""End-to-end frontend smoke test.

Runs the shipped Python path:

1. Build a FusedMLP model.
2. Capture the FX graph with ``forge_torch.compile``.
3. Confirm the ``forge.graph`` MLIR contains matmul, add, and gelu.
4. Run ``forge-opt --action=lower`` and ``--action=roundtrip`` on that
   MLIR with the current interpreter.
5. Validate the returned callable against eager and assert that the
   result names the eager execution path.
6. Assert the compilation report was emitted on the cold path and that
   its unmeasured fields stay ``None``.

The C++ runtime smoke is a separate test. It skips when the build tree
is absent, so the Python suite does not depend on a local CMake build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from forge_torch.backend import compile
from forge_torch.cache_store import CacheStore
from forge_torch.conftest import FusedMLP

REPO = Path(__file__).resolve().parents[2]
FORGE_OPT = REPO / "compiler" / "tools" / "forge-opt" / "forge_opt.py"
TEST_ABI_BIN = REPO / "build" / "runtime" / "tests" / "test_abi"


def _run_forge_opt(mlir_text: str, action: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FORGE_OPT), "-", f"--action={action}"],
        input=mlir_text,
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )


def test_phase1_smoke_full_pipeline(tmp_path: Path) -> None:
    """Run the Python slice and assert the labels match the work done."""
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(2, 8, 128)
    w = torch.randn(128, 256)
    b = torch.randn(256)

    compiled = compile(model, [x, w, b], cache_store=CacheStore(tmp_path / "cache"))

    mlir_text = compiled.mlir_text
    for op in ("forge.graph.matmul", "forge.graph.add", "forge.graph.gelu"):
        assert op in mlir_text, f"{op} missing from MLIR:\n{mlir_text[:2000]}"

    lower = _run_forge_opt(mlir_text, "lower")
    assert lower.returncode == 0, (
        f"forge-opt --action=lower failed (rc={lower.returncode}):\n"
        f"STDOUT:\n{lower.stdout[:2000]}\nSTDERR:\n{lower.stderr[:2000]}"
    )
    for op in (
        "forge.tensor.gemm",
        "forge.tensor.broadcast_add",
        "forge.tensor.gelu",
    ):
        assert op in lower.stdout, f"{op} missing from lowered MLIR:\n{lower.stdout[:2000]}"

    roundtrip = _run_forge_opt(mlir_text, "roundtrip")
    assert roundtrip.returncode == 0, (
        f"forge-opt --action=roundtrip failed (rc={roundtrip.returncode}):\n"
        f"STDOUT:\n{roundtrip.stdout}\nSTDERR:\n{roundtrip.stderr}"
    )

    eager_out = model(x, w, b)
    compiled_out = compiled(x, w, b)
    result = compiled.validate()
    expected_path = "native" if compiled.lowered else "eager"
    assert result.passed, result
    assert result.execution == expected_path
    assert result.lowered is compiled.lowered
    assert torch.allclose(compiled_out, eager_out, atol=1e-4, rtol=1e-4)

    assert compiled.report is not None, "cold compile must emit a report"
    assert compiled.report.execution_path == expected_path
    assert compiled.report.validation_max_error is not None
    if compiled.lowered:
        assert compiled.report.generated_kernels == 1
    else:
        assert compiled.report.generated_kernels is None


def test_cpp_runtime_abi_smoke() -> None:
    """Run the built ABI test when a local build tree exists."""
    if not TEST_ABI_BIN.exists():
        pytest.skip(
            f"test_abi binary not found at {TEST_ABI_BIN}; "
            "build the runtime with CMake to run this check"
        )
    completed = subprocess.run(
        [str(TEST_ABI_BIN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"test_abi failed (rc={completed.returncode}):\n"
        f"STDOUT:\n{completed.stdout[:2000]}\nSTDERR:\n{completed.stderr[:2000]}"
    )
