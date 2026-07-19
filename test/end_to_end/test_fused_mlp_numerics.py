"""FusedMLP numerical-correctness pytests (Phase 1, Task 13).

These tests pin down the :func:`forge_torch.validation.validate`
contract: identical tensors pass, shape mismatches fail, NaN is
detected, and the per-dtype tolerance table matches spec §15.
"""

from __future__ import annotations

import torch
from forge_torch.validation import TOLERANCES, validate


def test_validate_function_passes_on_identical_tensors() -> None:
    a = torch.randn(4, 8)
    b = a.clone()
    r = validate(a, b)
    assert r.passed
    assert r.max_abs_err == 0.0
    assert r.shape_ok and r.dtype_ok
    assert not r.has_nan and not r.has_inf


def test_validate_function_detects_shape_mismatch() -> None:
    a = torch.randn(4, 8)
    b = torch.randn(4, 16)
    r = validate(a, b)
    assert not r.passed
    assert not r.shape_ok
    assert any("shape" in e for e in r.errors)


def test_validate_function_detects_nan() -> None:
    a = torch.tensor([float("nan"), 1.0])
    b = torch.tensor([0.0, 1.0])
    r = validate(a, b)
    assert r.has_nan
    assert not r.passed


def test_validate_function_respects_tolerance_table() -> None:
    # Tolerances must match spec §15 / torch.allclose defaults.
    assert TOLERANCES[torch.float32]["atol"] == 1e-4
    assert TOLERANCES[torch.float32]["rtol"] == 1e-4
    assert TOLERANCES[torch.float16]["atol"] == 1e-2
    assert TOLERANCES[torch.float16]["rtol"] == 1e-2
    assert TOLERANCES[torch.bfloat16]["atol"] == 5e-2
    assert TOLERANCES[torch.bfloat16]["rtol"] == 5e-2
    # Three dtypes — fp32, fp16, bf16 — is the spec §15 contract.
    assert set(TOLERANCES) == {torch.float32, torch.float16, torch.bfloat16}
