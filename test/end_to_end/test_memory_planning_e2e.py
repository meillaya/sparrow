"""End-to-end structural check for FusedMLP memory planning (Task 30)."""

from __future__ import annotations

from itertools import combinations
from math import prod
from typing import TypedDict

import torch


class TensorJson(TypedDict):
    id: int
    name: str
    dtype: int
    shape: list[int]
    memory_space: int


class AllocationJson(TypedDict):
    tensor_id: int
    byte_offset: int
    byte_size: int
    alignment: int
    memory_space: int


class LaunchJson(TypedDict):
    grid: list[int]
    block: list[int]
    shared_memory_bytes: int


class NodeJson(TypedDict):
    id: int
    device: int
    artifact: int
    inputs: list[int]
    outputs: list[int]
    dependencies: list[int]
    launch: LaunchJson


class ExternalTensorJson(TypedDict):
    tensor_id: int
    external_name: str


class PlanJson(TypedDict):
    name: str
    tensors: list[TensorJson]
    allocations: list[AllocationJson]
    nodes: list[NodeJson]
    inputs: list[ExternalTensorJson]
    outputs: list[ExternalTensorJson]


def _tensor_bytes(tensor: TensorJson) -> int:
    return prod(tensor["shape"]) * 4


def _build_execution_plan(model: torch.nn.Module, shape: tuple[int, int, int, int]) -> PlanJson:
    batch, sequence, reduction, output = shape
    tensors: list[TensorJson] = [
        {
            "id": 0,
            "name": "x",
            "dtype": 0,
            "shape": [batch, sequence, reduction],
            "memory_space": 0,
        },
        {"id": 1, "name": "weight", "dtype": 0, "shape": [reduction, output], "memory_space": 0},
        {"id": 2, "name": "bias", "dtype": 0, "shape": [output], "memory_space": 0},
        {
            "id": 3,
            "name": "matmul",
            "dtype": 0,
            "shape": [batch, sequence, output],
            "memory_space": 0,
        },
        {
            "id": 4,
            "name": "gelu",
            "dtype": 0,
            "shape": [batch, sequence, output],
            "memory_space": 0,
        },
        {
            "id": 5,
            "name": "output",
            "dtype": 0,
            "shape": [batch, sequence, output],
            "memory_space": 0,
        },
    ]
    allocations: list[AllocationJson] = [
        {
            "tensor_id": tensor["id"],
            "byte_offset": index * 64,
            "byte_size": _tensor_bytes(tensor),
            "alignment": 64,
            "memory_space": tensor["memory_space"],
        }
        for index, tensor in enumerate(tensors)
    ]
    launch: LaunchJson = {"grid": [], "block": [], "shared_memory_bytes": 0}
    return {
        "name": model.__class__.__name__,
        "tensors": tensors,
        "allocations": allocations,
        "nodes": [
            {
                "id": 0,
                "device": 0,
                "artifact": 0,
                "inputs": [0, 1, 2],
                "outputs": [3],
                "dependencies": [],
                "launch": launch,
            },
            {
                "id": 1,
                "device": 0,
                "artifact": 1,
                "inputs": [3],
                "outputs": [4],
                "dependencies": [0],
                "launch": launch,
            },
            {
                "id": 2,
                "device": 0,
                "artifact": 2,
                "inputs": [4],
                "outputs": [5],
                "dependencies": [1],
                "launch": launch,
            },
        ],
        "inputs": [
            {"tensor_id": 0, "external_name": "x"},
            {"tensor_id": 1, "external_name": "weight"},
            {"tensor_id": 2, "external_name": "bias"},
        ],
        "outputs": [{"tensor_id": 5, "external_name": "output"}],
    }


def _plan_memory(plan: PlanJson, alignment: int = 64) -> None:
    """Mirror the C++ liveness-aware first-fit allocator until pybind11 lands."""
    sizes = {tensor["id"]: _tensor_bytes(tensor) for tensor in plan["tensors"]}
    live_ranges: dict[int, tuple[int, int]] = {}
    for tensor_id in sizes:
        uses = [
            node["id"]
            for node in plan["nodes"]
            if tensor_id in node["inputs"] or tensor_id in node["outputs"]
        ]
        live_ranges[tensor_id] = (min(uses, default=0), max(uses, default=0))

    ordered = sorted(
        sizes,
        key=lambda tensor_id: (
            -(live_ranges[tensor_id][1] - live_ranges[tensor_id][0]),
            -sizes[tensor_id],
            tensor_id,
        ),
    )
    total_bytes = sum(sizes.values())
    placed: list[AllocationJson] = []
    for tensor_id in ordered:
        byte_size = sizes[tensor_id]
        first_use, last_use = live_ranges[tensor_id]
        for byte_offset in range(0, total_bytes - byte_size + 1, alignment):
            conflict = False
            for allocation in placed:
                other_first, other_last = live_ranges[allocation["tensor_id"]]
                lifetimes_overlap = first_use <= other_last and other_first <= last_use
                bytes_overlap = (
                    byte_offset < allocation["byte_offset"] + allocation["byte_size"]
                    and allocation["byte_offset"] < byte_offset + byte_size
                )
                if lifetimes_overlap and bytes_overlap:
                    conflict = True
                    break
            if not conflict:
                placed.append(
                    {
                        "tensor_id": tensor_id,
                        "byte_offset": byte_offset,
                        "byte_size": byte_size,
                        "alignment": alignment,
                        "memory_space": 0,
                    }
                )
                break
        else:
            raise AssertionError(f"mock C++ memory pool cannot fit tensor {tensor_id}")

    plan["allocations"] = sorted(placed, key=lambda allocation: allocation["tensor_id"])


def test_fused_mlp_memory_plan_reuses_storage_without_aliasing_inputs(
    fused_mlp: torch.nn.Module,
) -> None:
    plan = _build_execution_plan(fused_mlp, (2, 4, 8, 16))

    _plan_memory(plan)

    required_fields = {"tensor_id", "byte_offset", "byte_size", "alignment", "memory_space"}
    assert all(set(allocation) == required_fields for allocation in plan["allocations"])
    peak = max(
        allocation["byte_offset"] + allocation["byte_size"] for allocation in plan["allocations"]
    )
    largest = max(allocation["byte_size"] for allocation in plan["allocations"])
    assert peak < 4 * largest

    allocations = {allocation["tensor_id"]: allocation for allocation in plan["allocations"]}
    for left_id, right_id in combinations((0, 1, 2), 2):
        left = allocations[left_id]
        right = allocations[right_id]
        assert (
            left["byte_offset"] + left["byte_size"] <= right["byte_offset"]
            or right["byte_offset"] + right["byte_size"] <= left["byte_offset"]
        )
