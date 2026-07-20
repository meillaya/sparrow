"""Phase 7 Config C end-to-end pytest (Task 23, spec §19).

Config C: elementwise / reduction ops route to Triton-CPU; MLP matmul
regions route to MicroGPU-sim. Mirrors spec §19 "elementwise on
Triton, MLP matmul on MicroGPU".

In the shipped function-style TransformerBlock mirror, every matmul
region (QKV / attention / out / MLP) routes to MicroGPU under
Config C — the simplest version of the spec §19 Config C mapping
routes every matmul to MicroGPU, and region-index refinement is a
later task. The test asserts:

  * every region assigned to a backend consistent with the spec §19
    Config C mapping (matmul → microgpu, non-matmul → triton)
  * compiled output matches eager within fp32 atol=1e-3
  * the spec §19 metric table fields are populated

The function-style model mirrors the spec §19 TransformerBlock topology
using only ops the shipped FX importer lowers today — see
``test_phase7_config_a.py`` for the rationale.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge_torch.compile_partitioned import (
    CompiledModel,
    compile_partitioned,
)
from forge_torch.partitioner import map_node_to_forge_op


# ---------------------------------------------------------------------------
# Function-style TransformerBlock mirror — duplicated from
# ``test_phase7_config_a.py`` so each test file is self-contained.
# ---------------------------------------------------------------------------


class _Spec19Transformer(nn.Module):
    """Spec §19 topology in FX-traceable form (matmul/add/gelu only)."""

    def forward(
        self,
        x: torch.Tensor,  # (B, S, d_model)
        w_qkv: torch.Tensor,
        b_qkv: torch.Tensor,
        w_attn: torch.Tensor,
        b_attn: torch.Tensor,
        w_out: torch.Tensor,
        b_out: torch.Tensor,
        w_mlp1: torch.Tensor,
        b_mlp1: torch.Tensor,
        w_mlp2: torch.Tensor,
        b_mlp2: torch.Tensor,
        w_ln1: torch.Tensor,
        b_ln1: torch.Tensor,
        w_ln2: torch.Tensor,
        b_ln2: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.matmul(x, w_ln1) + b_ln1  # LN1 approximation
        qkv = torch.matmul(h, w_qkv) + b_qkv
        attn = torch.matmul(qkv, w_attn) + b_attn
        x = x + torch.matmul(attn, w_out) + b_out
        h = torch.matmul(x, w_ln2) + b_ln2  # LN2 approximation
        h = torch.matmul(h, w_mlp1) + b_mlp1
        h = F.gelu(h)
        h = torch.matmul(h, w_mlp2) + b_mlp2
        x = x + h
        return x


D_MODEL = 64
N_HEADS = 4
D_FF = 128


def _make_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    B, S, D = 1, 4, D_MODEL
    d_ff = D_FF
    return [
        torch.randn(B, S, D, dtype=torch.float32),
        torch.randn(D, 3 * D, dtype=torch.float32),
        torch.randn(3 * D, dtype=torch.float32),
        torch.randn(3 * D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, d_ff, dtype=torch.float32),
        torch.randn(d_ff, dtype=torch.float32),
        torch.randn(d_ff, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
        torch.randn(D, D, dtype=torch.float32),
        torch.randn(D, dtype=torch.float32),
    ]


@pytest.fixture
def spec19_model() -> _Spec19Transformer:
    return _Spec19Transformer()


@pytest.fixture
def spec19_inputs() -> list[torch.Tensor]:
    return _make_inputs()


@pytest.fixture
def eager_output(
    spec19_model: _Spec19Transformer,
    spec19_inputs: list[torch.Tensor],
) -> torch.Tensor:
    spec19_model.eval()
    with torch.no_grad():
        return spec19_model(*spec19_inputs)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


_MATMUL_OPS = frozenset({"forge.tensor.gemm"})
_ELEMENTWISE_OPS = frozenset(
    {
        "forge.tensor.gelu",
        "forge.tensor.layer_norm",
        "forge.tensor.softmax",
        "forge.tensor.broadcast_add",
    }
)


def _region_op_kind(region: list) -> str:
    """Classify a region as 'matmul', 'elementwise', or 'mixed'."""
    op_names: set[str] = set()
    for node in region:
        mapped = map_node_to_forge_op(node)
        if mapped is not None:
            op_names.add(mapped)
    if not op_names:
        return "other"
    if op_names <= _MATMUL_OPS:
        return "matmul"
    if op_names & _MATMUL_OPS:
        return "mixed"
    if op_names & _ELEMENTWISE_OPS:
        return "elementwise"
    return "other"


# ---------------------------------------------------------------------------
# Task 23 acceptance: pytest exits 0; metrics captured; region→backend
# mapping matches spec §19 Config C (elementwise → triton, MLP matmul →
# microgpu).
# ---------------------------------------------------------------------------


def test_config_c_elementwise_to_triton_mlp_matmul_to_microgpu(
    spec19_model: _Spec19Transformer,
    spec19_inputs: list[torch.Tensor],
    eager_output: torch.Tensor,
) -> None:
    """Config C: elementwise → triton, matmul regions → microgpu."""
    t0 = time.perf_counter()
    compiled: CompiledModel = compile_partitioned(spec19_model, spec19_inputs, config="C")
    compile_time_ms = (time.perf_counter() - t0) * 1000.0

    # Per-region spec §19 Config C mapping. The shipped Config C routes
    # every matmul region to microgpu (the simplest version; region-
    # index refinement to isolate "MLP-only" matmul is a later task).
    for idx, (region, backend) in enumerate(zip(compiled.regions, compiled.region_assignments, strict=True)):
        kind = _region_op_kind(region)
        if kind == "matmul":
            assert backend == "microgpu", (
                f"region {idx} is a pure-matmul region; Config C must route it "
                f"to microgpu (got {backend!r})"
            )
        else:
            assert backend == "triton", (
                f"region {idx} is {kind!r}; Config C must route non-matmul "
                f"regions to triton (got {backend!r})"
            )

    # Both backends should appear in the assignment.
    assert "microgpu" in compiled.backend_set
    assert "triton" in compiled.backend_set

    t0 = time.perf_counter()
    with torch.no_grad():
        compiled_output = compiled(*spec19_inputs)
    latency_s = time.perf_counter() - t0

    numerical_error_max = (compiled_output - eager_output).abs().max().item()
    assert numerical_error_max < 1e-3, (
        f"Config C output diverges from eager by {numerical_error_max:.3e} "
        f"(fp32 atol=1e-3 required)"
    )

    # Spec §19 metric fields — all populated, non-NaN, non-negative.
    assert compile_time_ms >= 0.0
    assert latency_s >= 0.0
    assert numerical_error_max >= 0.0
    assert compiled.num_regions > 0
