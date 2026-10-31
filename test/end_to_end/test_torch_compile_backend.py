"""torch.compile integration through the Sparrow Dynamo backend."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("_sparrow_rt")

from forge_torch.dynamo import backend


class TwoLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def test_torch_compile_linear_model() -> None:
    torch.manual_seed(0)
    model = TwoLinear()
    x = torch.randn(2, 8)
    expected = model(x)

    compiled = torch.compile(model, backend=backend)
    actual = compiled(x)

    assert actual.shape == (2, 4)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
