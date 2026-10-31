"""The canonical OpSpec table is the single source of truth for op mappings."""

from __future__ import annotations

import torch

from forge_torch import ops
from forge_torch.capabilities import TRITON_CAPABILITIES
from forge_torch.partitioner import map_node_to_forge_op as partitioner_op
from forge_torch.region import map_node_to_forge_op as region_op


def test_specs_have_one_canonical_name_and_entry_point() -> None:
    names = [spec.op for spec in ops.OP_SPECS]
    assert len(names) == len(set(names))
    entry_points = [spec.entry_point for spec in ops.OP_SPECS if spec.entry_point]
    assert len(entry_points) == len(set(entry_points))


def test_aten_and_method_tables_are_derived_from_specs() -> None:
    for spec in ops.OP_SPECS:
        for name in spec.aten:
            assert ops.ATEN_OPS[name] is spec
        for name in spec.method:
            assert ops.METHOD_OPS[name] is spec


def test_normalize_target_handles_functions_and_overloads() -> None:
    assert ops.normalize_target(torch.matmul) == "aten.matmul"
    assert ops.normalize_target(torch.ops.aten.addmm.default) == "aten.addmm"
    assert ops.normalize_target("aten.transpose.int") == "aten.transpose"
    assert ops.normalize_target(torch.reshape) == "aten.reshape"


def test_supported_and_unsupported_sets_are_disjoint() -> None:
    assert not (set(ops.ATEN_OPS) & ops.UNSUPPORTED_BUT_RECOGNISED)


def test_recognised_unsupported_op_names_the_op() -> None:
    import pytest

    with pytest.raises(NotImplementedError, match="silu"):
        ops.spec_for_aten(torch.nn.functional.silu)


def test_region_and_partitioner_share_the_table() -> None:
    target = torch.ops.aten.relu.default
    assert ops.spec_for_aten(target).op == "forge.tensor.relu"
    # Both frontend views derive from ``ops.ATEN_OPS``.
    assert "aten.relu" in ops.ATEN_OPS
    assert callable(region_op)
    assert callable(partitioner_op)


def test_triton_capabilities_cover_every_lowerable_spec() -> None:
    # ``forge.tensor.copy`` is a plan-internal pass-through op, not an FX op.
    for spec in ops.OP_SPECS:
        if spec.entry_point is None or spec.op == "forge.tensor.copy":
            continue
        assert spec.op in TRITON_CAPABILITIES.supported_ops, spec.op


def test_cost_shapes_come_from_specs() -> None:
    assert ops.cost_shape_for("forge.tensor.gemm") == [8, 8, 8]
    assert ops.cost_shape_for("forge.tensor.transpose") == [8, 8]
    assert ops.cost_shape_for("forge.tensor.unknown") == [1, 1]
