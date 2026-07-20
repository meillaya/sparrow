"""Phase 7 CostModel pytest (Tasks 7-8, plan §Todos).

Pins the analytical cost the partitioner (Task 18) ranks candidate backends
against. Each test computes the expected cost by hand from the source-of-
truth files referenced in `cost_estimator.py` / `_phase7_cycle_tables.py`,
then asserts the cost estimator returns the same number. This guarantees
the partitioner's backend ranking is reproducible from the cycle tables
alone — no measured costs at this stage (spec §20 / plan §16).

Coverage (Task 8 acceptance = 4 tests; we ship exactly those 4):

  * Triton-CPU   — matmul (M*N*K cycles per the Path A triple-loop kernel).
  * Astra-sim    — microcode sequence (DMA_LOAD + MATMUL + VECTOR_GELU + DMA_STORE).
  * MicroGPU-sim — composed GELU (8 + 8 = 16 cycles flat per plan §16).
  * Cross-region — Triton → Astra transition (DMA-in bytes = output bytes).
"""

from __future__ import annotations

from frontend.python.forge_torch.cost_estimator import (
    CostEstimate,
    cross_region_cost,
    estimate_cost,
)

# ---------------------------------------------------------------------------
# Triton-CPU — naive triple-loop kernel (Path A reference).
# Mirrors `backends/triton/cpu/fused_gemm_bias_gelu.cpp`.
# ---------------------------------------------------------------------------


def test_triton_matmul_cycles() -> None:
    """Triton matmul on [M=128, N=128, K=128] = M * N * K = 2,097,152 cycles.

    The Path A kernel is a literal triple loop so the cycle count equals
    the number of multiply-adds performed. Mirrors Task 8(a) of the plan:
    the hand-computed value matches the cycle-table lambda
    ``shape[0] * shape[1] * shape[2]``.
    """
    region = [{"op": "forge.tensor.gemm", "shape": [128, 128, 128]}]
    estimate = estimate_cost(region, "triton")

    assert isinstance(estimate, CostEstimate)
    assert estimate.compute_cycles == 128 * 128 * 128
    assert estimate.compute_cycles == 2_097_152
    # Triton has no DMA model — bytes_transferred is always zero.
    assert estimate.bytes_transferred == 0


# ---------------------------------------------------------------------------
# Astra-sim — per-instruction cycle table from
# `backends/astra/sim/timing.py::_CYCLE_TABLE`.
# ---------------------------------------------------------------------------


def test_astra_microcode_sequence_cycles() -> None:
    """A 4-op Astra microcode sequence totals 28 cycles (4 + 16 + 4 + 4).

    DMA loads / stores charge 4 cycles per byte at ``shape[0] = 1``;
    matmul charges 16 cycles per 16x16 systolic-array tile (1 tile at
    ``shape = [1, 1, 1]``); vector_gelu charges 4 cycles per element at
    ``shape[0] = 1``. Mirrors Task 8(b) of the plan.
    """
    region = [
        {"op": "forge.microcode.dma_load", "shape": [1]},
        {"op": "forge.microcode.matmul", "shape": [1, 1, 1]},
        {"op": "forge.microcode.vector_gelu", "shape": [1]},
        {"op": "forge.microcode.dma_store", "shape": [1]},
    ]
    estimate = estimate_cost(region, "astra")

    assert estimate.compute_cycles == 4 + 16 + 4 + 4
    assert estimate.compute_cycles == 28
    # DMA bytes = 1-byte load + 1-byte store at shape[0] = 1.
    assert estimate.bytes_transferred == 2


# ---------------------------------------------------------------------------
# MicroGPU-sim — composed GELU = 2 x sigmoid = 8 + 8 = 16 cycles flat.
# Mirrors plan §16 / spec §20 ("gelu-like sequence gives 8+8=16 cycles").
# ---------------------------------------------------------------------------


def test_microgpu_gelu_sequence_cycles() -> None:
    """MicroGPU GELU is composed as 2 sigmoid invocations → 16 cycles flat.

    The shipped SIMT ISA has no native GELU primitive
    (`MICROGPU_SIM_CYCLES = {load/store_global, add, mul, sigmoid, return}`),
    so the cost model composes it from two sigmoid operations on the
    per-instruction cycle table. Mirrors Task 8(c) of the plan.
    """
    region = [{"op": "forge.tensor.gelu", "shape": [128, 128]}]
    estimate = estimate_cost(region, "microgpu")

    assert estimate.compute_cycles == 16
    # MicroGPU has no DMA model.
    assert estimate.bytes_transferred == 0


# ---------------------------------------------------------------------------
# Cross-region cost — Triton → Astra transition.
# ---------------------------------------------------------------------------


def test_cross_region_cost_triton_to_astra() -> None:
    """Cross-region cost Triton→Astra equals the Triton output tensor bytes.

    For a matmul region on ``[M, N, K]`` the output tensor is ``[M, N]``
    in fp32 = ``M * N * 4`` bytes. Mirrors Task 8(d).
    """
    region = [{"op": "forge.tensor.gemm", "shape": [128, 128, 128]}]
    bytes_xfer = cross_region_cost(region, "triton", "astra")

    # Triton matmul output = M * N * 4 = 128 * 128 * 4 = 65,536 bytes.
    assert bytes_xfer == 128 * 128 * 4
    assert bytes_xfer == 65_536
