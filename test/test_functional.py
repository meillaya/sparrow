"""Tests for the Astra functional simulator."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from backends.astra.sim.functional import (
    DRAM_BYTES,
    TILE_BYTES,
    TILE_DIM,
    last_dram_tensor,
    run_functional,
)
from compiler.lib.Dialect.Astra.assembler import assemble


def _gelu_reference(values: NDArray[np.float32]) -> NDArray[np.float32]:
    flat = np.fromiter(
        (
            np.float32(0.5 * float(value) * (1.0 + math.erf(float(value) / math.sqrt(2.0))))
            for value in values.flat
        ),
        dtype=np.float32,
        count=values.size,
    )
    return flat.reshape(values.shape)


def test_fused_mlp_matches_numpy_reference_within_fp32_atol() -> None:
    """Given a hand-written FusedMLP, execution matches NumPy."""
    k = 2
    rng = np.random.default_rng(50)
    lhs = rng.standard_normal((TILE_DIM, k)).astype(np.float32)
    rhs = rng.standard_normal((k, TILE_DIM)).astype(np.float32)
    lhs_bytes = lhs.nbytes
    rhs_bytes = rhs.nbytes
    rhs_sram = lhs_bytes
    rhs_dram = lhs_bytes
    gelu_sram = 2 * TILE_BYTES
    output_dram = lhs_bytes + rhs_bytes
    program = f"""\
DMA_LOAD dst=SRAM[0x0], src=DRAM[0x0], bytes={lhs_bytes}
DMA_LOAD dst=SRAM[0x{rhs_sram:x}], src=DRAM[0x{rhs_dram:x}], bytes={rhs_bytes}
MATMUL a=SRAM[0x0], b=SRAM[0x{rhs_sram:x}], k={k}
VECTOR_GELU dst=SRAM[0x{gelu_sram:x}], src=SRAM[0x0], elements={TILE_DIM * TILE_DIM}
DMA_STORE dst=DRAM[0x{output_dram:x}], src=SRAM[0x{gelu_sram:x}], bytes={TILE_BYTES}
HALT
"""
    dram = bytearray(DRAM_BYTES)
    dram[:lhs_bytes] = lhs.tobytes()
    dram[rhs_dram : rhs_dram + rhs_bytes] = rhs.tobytes()

    result = run_functional(assemble(program), dram=dram)

    output = last_dram_tensor(result)
    expected = _gelu_reference((lhs @ rhs).astype(np.float32))
    assert output is not None
    assert np.allclose(output, expected, atol=1e-4, rtol=0.0)
