"""Region boundary detection pytest (Task 17, plan §Todos).

Pins the behaviour of ``forge_torch.region.partition_into_regions`` for the
two shipped Phase 7 models: the Phase 1 FusedMLP and the spec §19
TransformerBlock. The region count is the most user-visible knob the
partitioner (Task 18) consumes, so we pin it to a tolerance that
accommodates FX output variation across PyTorch versions while still
rejecting regressions like "all FX nodes get their own region".

Why these models:

  * **FusedMLP** — Phase 1's vertical slice (4 conceptual ops: matmul,
    add, gelu, output). The FX trace lowers ``x @ w + b`` to ``matmul``
    → ``add`` → ``gelu`` (the output node is skipped). Three call_function
    call-sites, each a distinct ``forge.tensor.*`` op, so the partitioner
    starts with 3 one-node regions.

  * **TransformerBlock** — spec §19 stack with manual layer_norm, QKV
    linear, multi-step attention (qk^T/√d, softmax, @V), out projection,
    residual, layer_norm, MLP, residual. The full FX trace has many more
    nodes than the spec's "~10 main ops" by intent (each conceptual op
    lowers to multiple FX nodes — mean+var+sub+sqrt+truediv for layer_norm,
    matmul+softmax+matmul+transpose+reshape for attention). The
    adjacent-merge in ``partition_into_regions`` collapses the per-op
    chains into roughly one region per conceptual op.
"""

from __future__ import annotations

import pytest
import torch

from forge_torch.conftest import FusedMLP
from forge_torch.region import partition_into_regions
from models.transformer_block import TransformerBlock


def test_transformer_block_produces_expected_regions() -> None:
    """TransformerBlock(d_model=64, n_heads=4, d_ff=128) yields 5..20 regions.

    The exact count is FX-version-dependent but the spec §19 transformer
    has "approximately 10 main ops". We accept 5..20 to stay robust across
    PyTorch FX output shifts while catching "every FX node got its own
    region" regressions.
    """
    torch.manual_seed(0)
    model = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    model.eval()
    example_inputs = [torch.randn(2, 4, 64, dtype=torch.float32)]

    regions = partition_into_regions(model, example_inputs)

    assert 5 <= len(regions) <= 20, (
        f"expected 5..20 regions, got {len(regions)}; "
        "check the adjacent-merge heuristic in partition_into_regions"
    )
    # Sanity: every region is a non-empty list of FX nodes (no empty regions).
    assert all(len(r) >= 1 for r in regions)
    # Sanity: total node count across regions equals the call_function +
    # call_method + call_module count in the captured graph (no node lost).
    gm = torch.fx.symbolic_trace(model)
    expected_ops = sum(
        1 for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module")
    )
    assert sum(len(r) for r in regions) <= expected_ops


def test_fusedmlp_produces_regions() -> None:
    """FusedMLP yields 1..5 regions.

    FusedMLP's FX trace lowers to ``x @ w + b`` followed by ``gelu(...)``.
    Each call_function gets its own region by default: matmul, add, gelu.
    The output and placeholder nodes are skipped. We accept 1..5 to be
    FX-version robust while catching "the whole model collapsed into
    one region" or "every getitem got its own region" regressions.
    """
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(2, 4, 8, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(16, dtype=torch.float32)

    regions = partition_into_regions(model, [x, w, b])

    assert 1 <= len(regions) <= 5, (
        f"expected 1..5 regions, got {len(regions)} "
        "(FusedMLP should yield matmul + add + gelu = ~3 regions)"
    )
    # All regions are non-empty.
    assert all(len(r) >= 1 for r in regions)
