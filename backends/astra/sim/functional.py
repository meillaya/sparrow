"""Correctness-only Astra microcode simulator."""

# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, assert_never

import numpy as np
from numpy.typing import NDArray

from compiler.lib.Dialect.Astra.microcode import OpCode, decode

SRAM_BYTES: Final = 512 * 1024
DRAM_BYTES: Final = 64 * 1024 * 1024
TILE_DIM: Final = 16
TILE_BYTES: Final = TILE_DIM * TILE_DIM * np.dtype(np.float32).itemsize
_FLOAT_BYTES: Final = np.dtype(np.float32).itemsize
_INV_SQRT_TWO: Final = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class DramWrite:
    """One completed DMA write to DRAM."""

    offset: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class FunctionalResult:
    """Final memory state and ordered DRAM writes from one simulation."""

    dram: bytearray
    sram: bytearray
    dram_writes: tuple[DramWrite, ...]


def _gelu_fp32(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply the erf-based GELU formula without leaving float32."""
    scalars: array[float] = array("f")
    scalars.frombytes(values.tobytes())
    return np.fromiter(
        (np.float32(0.5 * value * (1.0 + math.erf(value * _INV_SQRT_TWO))) for value in scalars),
        dtype=np.float32,
        count=values.size,
    )


def run_functional(
    words: Sequence[int],
    *,
    dram: bytearray | None = None,
    sram: bytearray | None = None,
) -> FunctionalResult:
    """Execute packed Astra instructions without recording timing."""
    dram_buffer = bytearray(DRAM_BYTES) if dram is None else dram
    sram_buffer = bytearray(SRAM_BYTES) if sram is None else sram
    dram_writes: list[DramWrite] = []

    for word in words:
        instruction = decode(word)
        match instruction.opcode:
            case OpCode.DMA_LOAD:
                count = instruction.imm
                source = instruction.reg_b
                destination = instruction.reg_a
                sram_buffer[destination : destination + count] = dram_buffer[
                    source : source + count
                ]
            case OpCode.DMA_STORE:
                count = instruction.imm
                source = instruction.reg_b
                destination = instruction.reg_a
                payload = bytes(sram_buffer[source : source + count])
                dram_buffer[destination : destination + count] = payload
                dram_writes.append(DramWrite(offset=destination, payload=payload))
            case OpCode.MATMUL:
                k = instruction.imm
                lhs_count = TILE_DIM * k
                rhs_count = k * TILE_DIM
                lhs_start = instruction.reg_a
                rhs_start = instruction.reg_b
                lhs = np.frombuffer(
                    sram_buffer,
                    dtype=np.float32,
                    count=lhs_count,
                    offset=lhs_start,
                ).reshape(TILE_DIM, k)
                rhs = np.frombuffer(
                    sram_buffer,
                    dtype=np.float32,
                    count=rhs_count,
                    offset=rhs_start,
                ).reshape(k, TILE_DIM)
                product = (lhs @ rhs).astype(np.float32).tobytes()
                sram_buffer[lhs_start : lhs_start + TILE_BYTES] = product
            case OpCode.VECTOR_GELU:
                count = instruction.imm
                source = instruction.reg_b
                destination = instruction.reg_a
                values = np.frombuffer(
                    sram_buffer,
                    dtype=np.float32,
                    count=count,
                    offset=source,
                )
                payload = _gelu_fp32(values).tobytes()
                byte_count = count * _FLOAT_BYTES
                sram_buffer[destination : destination + byte_count] = payload
            case OpCode.NOP:
                pass
            case OpCode.HALT:
                break
            case unreachable:
                assert_never(unreachable)

    return FunctionalResult(
        dram=dram_buffer,
        sram=sram_buffer,
        dram_writes=tuple(dram_writes),
    )


def last_dram_tensor(result: FunctionalResult) -> NDArray[np.float32] | None:
    """Decode the final DMA store as a copied float32 tensor."""
    if not result.dram_writes:
        return None
    tensor = np.frombuffer(result.dram_writes[-1].payload, dtype=np.float32).copy()
    if tensor.size == TILE_DIM * TILE_DIM:
        return tensor.reshape(TILE_DIM, TILE_DIM)
    return tensor
