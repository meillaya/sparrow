"""Plan version 3 device tables and mixed CPU/Triton execution (M4)."""

from __future__ import annotations

import copy
import importlib
import json

import pytest
import torch
import torch.nn.functional as F

from compiler.lib.pipeline import lower_to_tensor
from forge_torch.fx_importer import FxImporter
from forge_torch.plan import CPU_DEVICE, TRITON_DEVICE, build_plan

pytest.importorskip("_sparrow_rt")
_sparrow_rt = importlib.import_module("_sparrow_rt")


class _FusedThenRelu(torch.nn.Module):
    def forward(self, x, w, b):
        return torch.relu(F.gelu(x @ w + b))


def _mixed_plan(module, inputs):
    gm = torch.fx.symbolic_trace(module)
    importer = FxImporter(inputs)
    lowered = lower_to_tensor(importer.import_module(gm))
    return build_plan(
        lowered,
        devices=[CPU_DEVICE, TRITON_DEVICE],
        fuse_triton=True,
    )


def _execute_json(plan_json, inputs):
    arrays = [tensor.detach().numpy() for tensor in inputs]
    outputs = _sparrow_rt.execute_plan(plan_json, arrays)
    return torch.from_numpy(outputs[0])


def _execute(plan, inputs):
    return _execute_json(plan.to_json(), inputs)


def test_mixed_cpu_and_triton_plan_executes() -> None:
    torch.manual_seed(0)
    model = _FusedThenRelu()
    x, w, b = torch.randn(4, 8), torch.randn(8, 6), torch.randn(6)
    plan = _mixed_plan(model, [x, w, b])

    assert plan.plan["version"] == 3
    assert [device["name"] for device in plan.plan["devices"]] == ["cpu", "triton"]
    targets = [artifact["target"] for artifact in plan.plan["artifacts"]]
    assert 4 in targets  # CPU
    assert 0 in targets  # Triton
    devices = {node["device"] for node in plan.plan["nodes"]}
    assert devices == {0, 1}

    actual = _execute(plan, [x, w, b])
    expected = model(x, w, b)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_version_2_plan_without_device_table_still_executes() -> None:
    """The scheduler keeps provider-index semantics for version 2 plans."""
    torch.manual_seed(0)
    model = _FusedThenRelu()
    x, w, b = torch.randn(4, 8), torch.randn(8, 6), torch.randn(6)
    plan = _mixed_plan(model, [x, w, b])

    legacy = copy.deepcopy(plan.plan)
    legacy["version"] = 2
    legacy.pop("devices")
    # Device ids 0/1 are also the provider order in the binding: CPU, Triton.
    actual = _execute_json(json.dumps(legacy), [x, w, b])
    expected = model(x, w, b)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
