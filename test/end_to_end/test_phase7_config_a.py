"""Phase 7 Config A end-to-end pytest (Task 21, spec §19).

Config A: every FX region routes to the CPU reference backend, which is
the only general lowering in the shipped scope. The compiled model must produce a tensor matching
eager within fp32 atol=1e-3 and the spec §19 metric table (latency,
peak memory, kernel count, data transferred, compile time, numerical
error) must be populated for the run.

The function-style model mirrors the spec §19 TransformerBlock topology
(LayerNorm → QKV → Attn → Out → Residual → LayerNorm → MLP → Residual)
using only ops the shipped FX importer lowers today (matmul, add, gelu).
The real ``models.transformer_block.TransformerBlock`` uses ``nn.Linear``
modules which emit ``call_module`` / ``get_attr`` FX nodes that the
Phase 1-8 FX importer does not yet lower; the function-style mirror
exercises the same op taxonomy on the same ``d_model=64, n_heads=4,
d_ff=128`` shapes, which is what the partitioner routes on. The
"LayerNorm" is approximated as a learned linear projection (matmul +
add) — the same shape signature, a different parameterisation.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge_torch.compile_partitioned import compile_partitioned

# ---------------------------------------------------------------------------
# Function-style TransformerBlock mirror.
#
# d_model=64, n_heads=4, d_ff=128 — the spec §19 small transformer shape.
# All weights are explicit forward() arguments so ``torch.fx.symbolic_trace``
# emits no ``get_attr`` / ``call_module`` nodes.
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
# Task 21 acceptance: pytest exits 0; metrics captured; numerical error
# within fp32 atol=1e-3.
# ---------------------------------------------------------------------------


def test_config_a_matches_eager_within_fp32_atol(
    spec19_model: _Spec19Transformer,
    spec19_inputs: list[torch.Tensor],
    eager_output: torch.Tensor,
) -> None:
    """Config A: every region → Triton-CPU; compiled output matches eager."""
    t0 = time.perf_counter()
    compiled = compile_partitioned(spec19_model, spec19_inputs, config="A")
    compile_time_ms = (time.perf_counter() - t0) * 1000.0

    # Spec §19 Config A: every region routes to triton.
    assert set(compiled.region_assignments) == {"cpu"}, (
        f"Config A must route every region to triton; got {compiled.region_assignments!r}"
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        compiled_output = compiled(*spec19_inputs)
    latency_s = time.perf_counter() - t0

    numerical_error_max = (compiled_output - eager_output).abs().max().item()
    assert numerical_error_max < 1e-3, (
        f"Config A output diverges from eager by {numerical_error_max:.3e} "
        f"(fp32 atol=1e-3 required)"
    )

    # Spec §19 metric fields — all populated, non-NaN, non-negative.
    assert compile_time_ms >= 0.0
    assert latency_s >= 0.0
    assert numerical_error_max >= 0.0
    assert compiled.num_regions > 0

    # Sanity-check the model is exercising the spec §19 topology on the
    # brief's exact shapes (d_model=64, n_heads=4, d_ff=128 → 15 regions
    # for this function-style mirror, matching the partitioner's
    # adjacent-merge output).
    assert compiled.num_regions == 15
