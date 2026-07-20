"""Phase 7 per-backend cycle-table pytest (Task 2, plan §Todos).

Pins the analytical cycle-cost contract the partitioner (Task 18) consumes.
Each test computes the expected cycle count by hand from the source-of-truth
files referenced in the module docstring, then asserts the table returns the
same number. The hand-computed values are intentionally small (M/N/K up to
128) so an arithmetic mistake in the lambda is obvious without a profiler.

Coverage map (Task 2 acceptance = 7+ tests; we ship 11):

  * Triton-CPU   — matmul (hand-computed M*N*K), gelu (M*N),
                   broadcast_add (M*N), layer_norm + softmax (M*N).
  * Astra-sim    — dma_load/store (4 * bytes), matmul (per-16x16-tile),
                   vector_gelu (4 * size).
  * MicroGPU-sim — load/store_global (4), add/mul (1), sigmoid (8),
                   return (0).
  * Phase 4 stub — read_phase4_stubs round-trip on a hand-built
                   CompilationReport.
"""

from __future__ import annotations

from frontend.python.forge_torch._phase7_cycle_tables import (
    ASTRA_SIM_CYCLES,
    MICROGPU_SIM_CYCLES,
    TRITON_CPU_CYCLES,
    read_phase4_stubs,
)
from frontend.python.forge_torch.report import CompilationReport

# ---------------------------------------------------------------------------
# Triton-CPU — naive triple-loop kernel (Path A reference).
# Mirrors `backends/triton/cpu/fused_gemm_bias_gelu.cpp`.
# ---------------------------------------------------------------------------


def test_triton_matmul_cycles_match_hand_computed_mnk() -> None:
    """Triton matmul on [M=128, N=128, K=128] = 128*128*128 = 2,097,152 cycles."""
    cycles = TRITON_CPU_CYCLES["matmul"]((128, 128, 128))
    assert cycles == 128 * 128 * 128
    assert cycles == 2_097_152


def test_triton_gelu_cycles_match_hand_computed_element_count() -> None:
    """Triton gelu on a (512, 4096) tensor touches 512*4096 = 2,097,152 elements."""
    cycles = TRITON_CPU_CYCLES["gelu"]((512, 4096))
    assert cycles == 512 * 4096
    assert cycles == 2_097_152


def test_triton_broadcast_add_layer_norm_softmax_share_output_size() -> None:
    """Elementwise ops (broadcast_add, layer_norm, softmax) all read M*N elements."""
    shape = (64, 128)
    expected = 64 * 128
    assert TRITON_CPU_CYCLES["broadcast_add"](shape) == expected
    assert TRITON_CPU_CYCLES["layer_norm"](shape) == expected
    assert TRITON_CPU_CYCLES["softmax"](shape) == expected


# ---------------------------------------------------------------------------
# Astra-sim — per-instruction cycle table from
# `backends/astra/sim/timing.py::_CYCLE_TABLE`. Matmul is per-16x16-tile.
# ---------------------------------------------------------------------------


def test_astra_dma_load_and_store_charge_four_cycles_per_byte() -> None:
    """DMA cycles scale linearly with bytes at 4 cycles/byte (timing.py)."""
    assert ASTRA_SIM_CYCLES["dma_load"](64) == 4 * 64
    assert ASTRA_SIM_CYCLES["dma_store"](32) == 4 * 32
    assert ASTRA_SIM_CYCLES["dma_load"](0) == 0  # zero-byte transfer is a no-op


def test_astra_matmul_charges_sixteen_cycles_per_16x16_tile() -> None:
    """Astra matmul on (M=64, N=64, K=64) yields 4*4*4=64 tiles * 16 cycles = 1024."""
    cycles_64 = ASTRA_SIM_CYCLES["matmul"]((64, 64, 64))
    assert cycles_64 == 4 * 4 * 4 * 16
    assert cycles_64 == 1024

    # (32, 32, 32) → 2*2*2 = 8 tiles x 16 cycles = 128.
    cycles_32 = ASTRA_SIM_CYCLES["matmul"]((32, 32, 32))
    assert cycles_32 == 8 * 16
    assert cycles_32 == 128


def test_astra_matmul_rounds_partial_tiles_up() -> None:
    """A dim not on the 16-tile boundary still costs one full extra tile."""
    # (17, 17, 17) → ceil(17/16) = 2 per axis → 2*2*2 = 8 tiles x 16 = 128.
    cycles = ASTRA_SIM_CYCLES["matmul"]((17, 17, 17))
    assert cycles == 128


def test_astra_vector_gelu_charges_four_cycles_per_element() -> None:
    """vector_gelu cost scales linearly with element count (timing.py)."""
    assert ASTRA_SIM_CYCLES["vector_gelu"](512) == 4 * 512
    assert ASTRA_SIM_CYCLES["vector_gelu"](1) == 4
    assert ASTRA_SIM_CYCLES["vector_gelu"](0) == 0


# ---------------------------------------------------------------------------
# MicroGPU-sim — per-instruction cycles from `backends/microgpu/interp.py`.
# ---------------------------------------------------------------------------


def test_microgpu_global_load_and_store_each_cost_four_cycles() -> None:
    """Per `interp.py`: LOAD_GLOBAL=4, STORE_GLOBAL=4 (i.e. memory ops)."""
    assert MICROGPU_SIM_CYCLES["load_global"] == 4
    assert MICROGPU_SIM_CYCLES["store_global"] == 4


def test_microgpu_arith_ops_match_per_instruction_cycles() -> None:
    """Per `interp.py`: ADD=1, MUL=1, SIGMOID=8, RETURN=0."""
    assert MICROGPU_SIM_CYCLES["add"] == 1
    assert MICROGPU_SIM_CYCLES["mul"] == 1
    assert MICROGPU_SIM_CYCLES["sigmoid"] == 8
    # RETURN terminates the program; it costs zero extra cycles beyond what
    # was already dispatched, so the partitioner can treat it as a no-op.
    assert MICROGPU_SIM_CYCLES["return"] == 0


# ---------------------------------------------------------------------------
# Phase 4 cost-stub reader (Task 2 + Task 9 of the plan).
# ---------------------------------------------------------------------------


def test_read_phase4_stubs_round_trips_alternative_cost_strings() -> None:
    """`read_phase4_stubs` mirrors the CompilationReport cost fields by name."""
    report = CompilationReport(
        model_name="FusedMLP",
        alternative_astra_cost="1.31x",
        alternative_fpga_cost="2.04x including transfer",
    )

    stubs = read_phase4_stubs(report)

    assert stubs == {
        "alternative_astra_cost": "1.31x",
        "alternative_fpga_cost": "2.04x including transfer",
    }
    # The keys are the dataclass field names verbatim so callers can index
    # by attribute name without translation.
    assert stubs["alternative_astra_cost"] == report.alternative_astra_cost
    assert stubs["alternative_fpga_cost"] == report.alternative_fpga_cost


def test_read_phase4_stubs_handles_empty_report_without_error() -> None:
    """A freshly-instantiated CompilationReport yields empty strings (not KeyError)."""
    report = CompilationReport(model_name="<unset>")
    stubs = read_phase4_stubs(report)

    assert stubs == {
        "alternative_astra_cost": "",
        "alternative_fpga_cost": "",
    }
