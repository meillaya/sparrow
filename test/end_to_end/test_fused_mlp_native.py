"""Native FusedMLP path.

Skips when the optional ``_sparrow_rt`` extension is not built. The test
checks that the callable is marked lowered, that validation compares a
real kernel output to eager, and that the numerical error is within the
documented fp32 tolerance.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("_sparrow_rt", reason="native bridge is not built")

from forge_torch.backend import compile
from forge_torch.conftest import FusedMLP


def test_native_fused_mlp_matches_eager() -> None:
    model = FusedMLP()
    x = torch.randn(2, 8, 128, dtype=torch.float32)
    w = torch.randn(128, 256, dtype=torch.float32)
    b = torch.randn(256, dtype=torch.float32)

    compiled = compile(model, [x, w, b])

    assert compiled.lowered is True
    assert compiled.execution == "native"
    out = compiled(x, w, b)
    assert out.shape == (2, 8, 256)
    assert torch.allclose(out, model(x, w, b), atol=1e-4, rtol=1e-4)

    result = compiled.validate(atol=1e-4, rtol=1e-4)
    assert result.passed
    assert result.lowered is True
    assert result.execution == "native"
    assert result.max_error <= 1e-4
