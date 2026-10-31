"""General CPU plan execution through compile()."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("_sparrow_rt")

from forge_torch.backend import compile
from forge_torch.cache_store import CacheStore


class TwoLayer(torch.nn.Module):
    def forward(self, x, w1, b1, w2, b2):
        hidden = torch.nn.functional.gelu(x @ w1 + b1)
        return hidden @ w2 + b2


def _inputs() -> list[torch.Tensor]:
    return [
        torch.randn(2, 8, 128),
        torch.randn(128, 256),
        torch.randn(256),
        torch.randn(256, 64),
        torch.randn(64),
    ]


def test_general_graph_executes_through_cpu_plan() -> None:
    torch.manual_seed(0)
    model = TwoLayer()
    inputs = _inputs()

    compiled = compile(model, inputs)
    actual = compiled(*inputs)
    expected = model(*inputs)

    assert compiled.lowered is True
    assert compiled.execution == "cpu"
    assert compiled.planning_error is None
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert actual.shape == (2, 8, 64)


def test_validation_reports_real_cpu_error(tmp_path) -> None:
    torch.manual_seed(0)
    model = TwoLayer()
    compiled = compile(model, _inputs(), cache_store=CacheStore(tmp_path / "cache"))

    result = compiled.validate()

    assert result.lowered is True
    assert result.execution == "cpu"
    assert result.passed is True
    assert result.max_error > 0.0
    assert compiled.report is not None
    assert compiled.report.validation_max_error == result.max_error


def test_plan_cache_round_trips(tmp_path) -> None:
    torch.manual_seed(0)
    model = TwoLayer()
    inputs = _inputs()
    store = CacheStore(tmp_path / "cache")
    first = compile(model, inputs, cache_store=store)
    second = compile(model, inputs, cache_store=store)

    assert second.cache_hit is True
    assert second.execution == "cpu"
    assert torch.allclose(first(*inputs), second(*inputs), atol=1e-6, rtol=1e-6)


def test_plan_rejects_wrong_input_shape() -> None:
    torch.manual_seed(0)
    model = TwoLayer()
    inputs = _inputs()
    compiled = compile(model, inputs)

    with pytest.raises(ValueError, match="shape"):
        compiled(inputs[0][:1], *inputs[1:])


class LinearStack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.nn.functional.gelu(self.fc1(x))))


def test_module_parameters_are_lifted_into_the_plan(tmp_path) -> None:
    torch.manual_seed(0)
    model = LinearStack().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)

    compiled = compile(model, [x], cache_store=CacheStore(tmp_path / "cache"))
    with torch.no_grad():
        actual = compiled(x)

    assert compiled.execution == "cpu"
    assert compiled.planning_error is None
    assert len(compiled._lifted_inputs) == 4
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
