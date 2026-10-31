"""Plan builder tests for the CPU execution path."""

from __future__ import annotations

import json

import pytest
import torch

from compiler.lib.pipeline import lower_to_tensor
from forge_torch.conftest import FusedMLP
from forge_torch.fx_importer import FxImporter
from forge_torch.plan import (
    UnsupportedDtypeError,
    UnsupportedOpError,
    build_plan,
)


class Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _module(model: torch.nn.Module, inputs: list[torch.Tensor]):
    gm = torch.fx.symbolic_trace(model)
    module = FxImporter(inputs).import_module(gm)
    return module


def test_fused_mlp_plan_has_one_node_per_tensor_op() -> None:
    inputs = [torch.randn(2, 8, 128), torch.randn(128, 256), torch.randn(256)]
    module = _module(FusedMLP(), inputs)
    lower_to_tensor(module)

    built = build_plan(module)

    assert [node["artifact"] for node in built.plan["nodes"]] == [0, 1, 2]
    assert [artifact["entry_point"] for artifact in built.plan["artifacts"]] == [
        "forge.tensor.gemm",
        "forge.tensor.broadcast_add",
        "forge.tensor.gelu",
    ]
    assert built.plan["nodes"][1]["dependencies"] == [0]
    assert built.plan["nodes"][2]["dependencies"] == [1]
    assert len(built.inputs) == 3
    assert built.outputs[0].shape == (2, 8, 256)
    json.loads(built.to_json())


def test_identity_output_gets_a_copy_node() -> None:
    module = _module(Identity(), [torch.randn(4, 8)])
    lower_to_tensor(module)

    built = build_plan(module)

    assert len(built.plan["nodes"]) == 1
    assert built.plan["artifacts"][0]["entry_point"] == "forge.tensor.copy"
    assert built.plan["inputs"][0]["tensor_id"] == 0
    assert built.plan["outputs"][0]["tensor_id"] == 1


def test_unlowered_module_is_rejected() -> None:
    module = _module(FusedMLP(), [torch.randn(2, 8, 128), torch.randn(128, 256), torch.randn(256)])

    with pytest.raises(UnsupportedOpError, match=r"forge\.graph\.matmul"):
        build_plan(module)


def test_fp16_is_rejected() -> None:
    inputs = [
        torch.randn(2, 8, 128, dtype=torch.float16),
        torch.randn(128, 256, dtype=torch.float16),
        torch.randn(256, dtype=torch.float16),
    ]
    module = _module(FusedMLP(), inputs)
    lower_to_tensor(module)

    with pytest.raises(UnsupportedDtypeError, match="fp32"):
        build_plan(module)


class MovementOps(torch.nn.Module):
    def forward(self, x, scale):
        y = x.permute(2, 0, 1)  # 2x3x4 -> 4x2x3
        z = y.reshape(4, 6)
        row = z[1]  # 6
        return torch.relu(row * scale) + row


def test_movement_ops_become_plan_nodes() -> None:
    x = torch.randn(2, 3, 4)
    scale = torch.randn(6)
    module = _module(MovementOps(), [x, scale])
    lower_to_tensor(module)

    built = build_plan(module)

    entry_points = [artifact["entry_point"] for artifact in built.plan["artifacts"]]
    for expected in (
        "forge.tensor.transpose",
        "forge.tensor.reshape",
        "forge.tensor.select",
        "forge.tensor.mul",
        "forge.tensor.relu",
        "forge.tensor.broadcast_add",
    ):
        assert expected in entry_points, entry_points

    transpose = next(
        artifact
        for artifact in built.plan["artifacts"]
        if artifact["entry_point"] == "forge.tensor.transpose"
    )
    attributes = json.loads(bytes(transpose["binary"]).decode())
    assert attributes == {"permutation": [2, 0, 1]}
    json.loads(built.to_json())


class AddmmBmm(torch.nn.Module):
    def forward(self, bias, a, b, d):
        x = torch.addmm(bias, a, b)  # (4, 5)
        return torch.bmm(x.reshape(1, 4, 5), d)  # (1, 4, 6)


def test_addmm_and_bmm_become_gemm_nodes() -> None:
    bias, a, b = torch.randn(5), torch.randn(4, 8), torch.randn(8, 5)
    d = torch.randn(1, 5, 6)
    module = _module(AddmmBmm(), [bias, a, b, d])
    lower_to_tensor(module)

    built = build_plan(module)

    artifacts = built.plan["artifacts"]
    node_entry_points = [artifacts[node["artifact"]]["entry_point"] for node in built.plan["nodes"]]
    assert node_entry_points.count("forge.tensor.gemm") == 2
    assert "forge.tensor.broadcast_add" in node_entry_points
    assert "forge.tensor.reshape" in node_entry_points
    assert built.outputs[0].shape == (1, 4, 6)
