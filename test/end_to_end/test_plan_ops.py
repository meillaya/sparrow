"""End-to-end execution of the M3 op set through a CPU plan."""

from __future__ import annotations

import importlib

import pytest
import torch

from compiler.lib.pipeline import lower_to_tensor
from forge_torch.fx_importer import FxImporter
from forge_torch.plan import build_plan

pytest.importorskip("_sparrow_rt")
_sparrow_rt = importlib.import_module("_sparrow_rt")


def _execute(model: torch.nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    gm = torch.fx.symbolic_trace(model)
    importer = FxImporter(inputs)
    module = importer.import_module(gm)
    lower_to_tensor(module)
    plan = build_plan(module)

    arrays = [tensor.detach().numpy() for tensor in inputs]
    arrays.extend(tensor.detach().numpy() for tensor in importer.lifted_tensors)
    outputs = _sparrow_rt.execute_plan(plan.to_json(), arrays)
    assert len(outputs) == 1
    return torch.from_numpy(outputs[0])


class ElementwiseMovement(torch.nn.Module):
    def forward(self, x, scale):
        y = x.permute(2, 0, 1)  # 2x3x4 -> 4x2x3
        z = y.reshape(4, 6)
        row = z[1]  # 6
        return torch.relu(row * scale) + row


def test_movement_and_elementwise_plan_matches_eager() -> None:
    torch.manual_seed(0)
    model = ElementwiseMovement()
    x = torch.randn(2, 3, 4)
    scale = torch.randn(6)

    actual = _execute(model, [x, scale])
    expected = model(x, scale)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


class MatmulChain(torch.nn.Module):
    def forward(self, bias, a, b, d):
        x = torch.addmm(bias, a, b)  # 4x5
        return torch.bmm(x.reshape(1, 4, 5), d).reshape(4, 6)


def test_addmm_and_bmm_plan_matches_eager() -> None:
    torch.manual_seed(0)
    model = MatmulChain()
    bias, a, b = torch.randn(5), torch.randn(4, 8), torch.randn(8, 5)
    d = torch.randn(1, 5, 6)

    actual = _execute(model, [bias, a, b, d])
    expected = model(bias, a, b, d)

    assert actual.shape == (4, 6)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
