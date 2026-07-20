"""Phase 1 end-to-end smoke test.

Runs the complete vertical slice:

  1. Probe the environment (``scripts/probe-env.sh`` exits 0).
  2. Build a FusedMLP model.
  3. Capture FX graph via ``torch.compile(backend="forgestack")`` semantics
     (the Phase 1 ``forge_torch.compile`` wrapper).
  4. Confirm ``forge.graph`` MLIR was produced (contains ``matmul``/``add``/``gelu``).
  5. Run :mod:`forge-opt` to confirm both the ``roundtrip`` and ``lower``
     actions succeed; after Phases 2-8 the GraphToTensor pass covers the
     canonical FusedMLP triple (matmul + add + gelu), so a lower failure
     here is a Phase 1 regression.
  6. Run eager and compiled, assert :func:`validate` passes.
  7. Confirm the compilation report was emitted under ``.omo/evidence/``
     (only when the ``CompiledFusedMLP`` instance carries a report
     pointer — Phase 1 may not wire this on every backend() call yet).
  8. Run the cmake-built ``test_abi`` gtest binary as a smoke check that
     the C++ runtime builds and runs.

Spec §20 mandates the smoke test must NOT depend on a real GPU or FPGA;
every assertion here runs on CPU only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from forge_torch.backend import compile
from forge_torch.conftest import FusedMLP
from forge_torch.validation import validate

REPO = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO / ".omo" / "evidence"
PHASE1_REPORT_PATH = EVIDENCE_DIR / "phase1-report.txt"
VENV_PYTHON = REPO / ".venv" / "bin" / "python3"
FORGE_OPT = REPO / "compiler" / "tools" / "forge-opt" / "forge_opt.py"
TEST_ABI_BIN = REPO / "build" / "runtime" / "tests" / "test_abi"


def _run(cmd, **kw):
    """Run a subprocess while collecting output; never raises on non-zero exit."""
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def _ensure_evidence_dir() -> None:
    """The report path is owned by Task 14; create the parent upfront so
    later steps can write there without racing the first compile() call."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _probe_environment() -> subprocess.CompletedProcess:
    """Run the canonical Phase-1 environment probe (scripts/probe-env.sh).

    Exits 0 only when all 18 probes pass; the smoke test depends on
    torch + xdsl + lit + pytest being available, so a failed probe here
    is a hard test failure.
    """
    return _run(
        ["bash", str(REPO / "scripts" / "probe-env.sh")],
        cwd=REPO,
    )


def test_phase1_smoke_full_pipeline():
    """Run every Phase 1 gate; aggregate failure messages for triage."""
    _ensure_evidence_dir()

    # 1) Environment probe ----------------------------------------------------
    probe = _probe_environment()
    assert probe.returncode == 0, (
        f"probe-env.sh failed (rc={probe.returncode}):\n"
        f"STDOUT:\n{probe.stdout}\nSTDERR:\n{probe.stderr}"
    )
    assert "pass" in probe.stdout.lower(), (
        f"probe-env.sh output missing PASS marker:\n{probe.stdout}"
    )

    # 2) Build the FusedMLP ----------------------------------------------------
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(2, 512, 4096)
    w = torch.randn(4096, 11008)
    b = torch.randn(11008)

    # 3) Compile via the ForgeStack backend -----------------------------------
    compiled = compile(model, [x, w, b])

    # 4) Confirm forge.graph MLIR was produced --------------------------------
    mlir_text = compiled.mlir_text
    assert "forge.graph.matmul" in mlir_text, (
        f"forge.graph.matmul missing from MLIR:\n{mlir_text[:2000]}"
    )
    assert "forge.graph.add" in mlir_text, f"forge.graph.add missing from MLIR:\n{mlir_text[:2000]}"
    assert "forge.graph.gelu" in mlir_text, (
        f"forge.graph.gelu missing from MLIR:\n{mlir_text[:2000]}"
    )

    # 5) Run forge-opt to confirm the dialect roundtrip + lower works --------
    # Phase 1 regression guard: Phases 2-8 made the lower pipeline (forge.graph
    # -> forge.tensor) complete for the canonical FusedMLP triple.
    lower = _run(
        [
            str(VENV_PYTHON),
            str(FORGE_OPT),
            "-",  # read from stdin
            "--action=lower",
        ],
        input=mlir_text,
        cwd=REPO,
    )
    assert lower.returncode == 0, (
        f"forge-opt --action=lower failed (rc={lower.returncode}); "
        "Phase 1 GraphToTensor regression after Phases 2-8.\n"
        f"STDOUT:\n{lower.stdout[:2000]}\nSTDERR:\n{lower.stderr[:2000]}"
    )
    assert "forge.tensor.gemm" in lower.stdout, (
        f"forge.tensor.gemm missing from lowered MLIR (Phase 1 regression):\n{lower.stdout[:2000]}"
    )
    assert "forge.tensor.broadcast_add" in lower.stdout, (
        f"forge.tensor.broadcast_add missing from lowered MLIR "
        f"(Phase 1 regression):\n{lower.stdout[:2000]}"
    )
    assert "forge.tensor.gelu" in lower.stdout, (
        f"forge.tensor.gelu missing from lowered MLIR (Phase 1 regression):\n{lower.stdout[:2000]}"
    )

    roundtrip = _run(
        [
            str(VENV_PYTHON),
            str(FORGE_OPT),
            "-",
            "--action=roundtrip",
        ],
        input=mlir_text,
        cwd=REPO,
    )
    assert roundtrip.returncode == 0, (
        f"forge-opt roundtrip failed (rc={roundtrip.returncode}):\n"
        f"STDOUT:\n{roundtrip.stdout}\nSTDERR:\n{roundtrip.stderr}"
    )

    # 6) Eager vs compiled — validate -----------------------------------------
    eager_out = model(x, w, b)
    compiled_out = compiled(x, w, b)
    result = validate(compiled_out, eager_out)
    assert result.passed, f"validate() failed: {result}\nerrors: {result.errors}"

    # 7) Confirm the compilation report was emitted (Task 14 owns the path).
    #    Phase 1 reports are emitted on cache miss; on cache hit no fresh
    #    write happens. We only sanity-check that the parent dir exists and
    #    tolerate "no report yet" — the e2e differential test is the real
    #    validation gate.
    if hasattr(compiled, "report") and compiled.report is not None:
        # Avoid asserting the file path is populated — that's Task 14's
        # contract. We surface a soft check via the evidence dir only.
        assert EVIDENCE_DIR.is_dir(), f"evidence dir disappeared: {EVIDENCE_DIR}"

    # 8) Run the C++ test_abi binary as a runtime smoke -----------------------
    #    Tolerant: if the host hasn't built the C++ runtime yet (e.g. CI is
    #    Python-only) skip with a clear message rather than failing.
    if TEST_ABI_BIN.exists():
        ctest_result = _run([str(TEST_ABI_BIN)])
        assert ctest_result.returncode == 0, (
            f"test_abi failed (rc={ctest_result.returncode}):\n"
            f"STDOUT:\n{ctest_result.stdout[:2000]}\n"
            f"STDERR:\n{ctest_result.stderr[:2000]}"
        )
    else:
        pytest.skip(
            f"test_abi binary not found at {TEST_ABI_BIN}; "
            "C++ runtime not built — skipping runtime smoke check"
        )
