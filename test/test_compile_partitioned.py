"""Heterogeneous ``compile_partitioned`` pytest (Task 19, plan §Todos).

Pins the Phase 7 heterogeneous compile entry point for all three
spec §19 configurations. Each test:

  1. Builds a SIMPLIFIED ``nn.Module`` whose ``forward()`` takes weights
     as explicit arguments (function-style — no ``nn.Parameter`` so
     ``torch.fx.symbolic_trace`` does not emit ``get_attr`` nodes, which
     the FX importer does not yet support).
  2. Runs eager forward to capture the reference output.
  3. Calls ``compile_partitioned(model, inputs, config=...)`` and
     invokes the returned :class:`CompiledModel` on the same inputs.
  4. Asserts the compiled output matches eager within ``atol=1e-3``
     (fp32 loose tolerance — the 3 backends may differ in the order of
     reduction per spec §19).

The simplified model uses only ops in
``forge_torch.op_mapping.SUPPORTED_ATEN_OPS`` so the FX importer
(:func:`forge_torch.backend.compile`) succeeds. The op set:

  * ``aten.matmul``         → ``forge.tensor.gemm``
  * ``aten.add``            → ``forge.tensor.broadcast_add``
  * ``aten.gelu`` (F.gelu)  → ``forge.tensor.gelu``
  * ``aten.softmax`` (F.softmax) → ``forge.tensor.softmax``

These exercise the four op classes (matmul, broadcast_add, gelu,
softmax) the spec §19 mappings for Config A/B/C differentiate.

Why not the full TransformerBlock (Task 16)?
--------------------------------------------
The shipped TransformerBlock uses ``nn.Linear`` modules which lower to
``call_module`` ops in FX; the Phase 7 FX importer
(``forge_torch.fx_importer``) does not yet support ``call_module``
because doing so requires ``get_attr`` handling for the submodule's
parameters (Phase 1-8 territory). The simplified model is a stand-in
that exercises the same op classes on smaller shapes; Task 19's
acceptance is "each config returns a tensor matching eager".
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge_torch.compile_partitioned import compile_partitioned

# ---------------------------------------------------------------------------
# Shared simplified model + helper.
#
# The model exercises 4 of the 4 elementwise + matmul op classes
# (matmul, broadcast_add, gelu, softmax). Weights are passed as explicit
# arguments so ``torch.fx.symbolic_trace`` produces no ``get_attr`` nodes
# — the FX importer emits each call via ``forge.tensor.<op>``.
# ---------------------------------------------------------------------------


class _SimpleLayeredModel(nn.Module):
    """Three matmul-add-gelu/softmax layers on a function-style input.

    Layer structure (deliberately minimal so every op class appears at
    least once for the partitioner to differentiate):

        layer 1: matmul → add → gelu
        layer 2: matmul → add → softmax
        layer 3: matmul → add

    The partitioner (Task 18) sees 8 regions: ``[gemm, broadcast_add,
    gelu, gemm, broadcast_add, softmax, gemm, broadcast_add]``. That
    gives Config A (all → triton), B (matmul regions → astra,
    elementwise → triton), and C (matmul regions → microgpu,
    elementwise → triton) all a non-trivial assignment to verify.
    """

    def forward(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        w3: torch.Tensor,
        b3: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.matmul(x, w1) + b1
        h = F.gelu(h)
        h = torch.matmul(h, w2) + b2
        h = F.softmax(h, dim=-1)
        h = torch.matmul(h, w3) + b3
        return h


def _make_inputs(seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [
        torch.randn(2, 4, 8, dtype=torch.float32),
        torch.randn(8, 16, dtype=torch.float32),
        torch.randn(16, dtype=torch.float32),
        torch.randn(16, 16, dtype=torch.float32),
        torch.randn(16, dtype=torch.float32),
        torch.randn(16, 8, dtype=torch.float32),
        torch.randn(8, dtype=torch.float32),
    ]


@pytest.fixture
def simple_model() -> _SimpleLayeredModel:
    return _SimpleLayeredModel()


@pytest.fixture
def simple_inputs() -> list[torch.Tensor]:
    return _make_inputs()


@pytest.fixture
def eager_output(
    simple_model: _SimpleLayeredModel, simple_inputs: list[torch.Tensor]
) -> torch.Tensor:
    simple_model.eval()
    with torch.no_grad():
        return simple_model(*simple_inputs)


# ---------------------------------------------------------------------------
# Config A is the only runnable configuration: it builds a CPU plan.
# ---------------------------------------------------------------------------


def test_config_a_builds_cpu_plan_and_matches_eager(
    simple_model: _SimpleLayeredModel,
    simple_inputs: list[torch.Tensor],
    eager_output: torch.Tensor,
) -> None:
    """Config A (``"A"``) runs the CPU execution plan, not the FX interpreter."""
    compiled = compile_partitioned(simple_model, simple_inputs, config="A")

    assert set(compiled.region_assignments) == {"cpu"}
    assert compiled.execution == "cpu"
    assert compiled.lowered is True
    assert compiled.planning_only is False

    compiled_output = compiled(*simple_inputs)
    assert torch.allclose(compiled_output, eager_output, atol=1e-3), (
        f"Config A output diverges from eager: max abs diff = "
        f"{(compiled_output - eager_output).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Configs B and C fail closed until their lowerings land.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "backend"),
    [("B", "Astra"), ("C", "MicroGPU")],
)
def test_unlowered_configs_raise(config: str, backend: str, simple_model, simple_inputs) -> None:
    with pytest.raises(NotImplementedError, match=backend):
        compile_partitioned(simple_model, simple_inputs, config=config)
