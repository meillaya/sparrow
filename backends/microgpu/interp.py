"""MicroGPU SIMT bytecode interpreter (spec §11).

The interpreter is a tiny reference implementation: a single in-order
fetch/decode/execute loop over a 32-slot register file, with the
backing store carried in a caller-provided ``bytearray``. It is
deliberately small — the goal is "reference correctness" for the
assembler and dialect, not high-throughput simulation. The Python
reference is also what the C++ :class:`forge_rt::MicroGpuProvider`
ultimately delegates to (via subprocess), so the test surface lives
here.

Tasks 76/77/78 layer race detection, out-of-bounds detection, and
the bytecode dep-DAG emitter on top of this file's run loop. The run
function returns the final register file so the host (test, driver,
provider) can inspect the post-state.

Reference: spec §11.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from backends.microgpu.isa import OpCode, decode


# 32-register file (r0..r31). The wire format only encodes the low 16
# slots (4-bit register field), so an instruction can only read/write
# r0..r15; the higher 16 slots are reachable via the host's pre-loaded
# ``regs`` argument (used by the host-side launch path to pass
# scalar arguments into the warp). The runtime provider uses both.
REGISTER_COUNT: Final = 32

# Bytes per memory word (every LOAD/STORE moves exactly one i32).
BYTES_PER_WORD: Final = 4

# The sigmoid we expose is the standard logistic 1/(1+e^-x), which
# matches the spec's ``SIGMOID`` op definition. The task uses this
# function for both the reference numpy comparison and the runtime
# behaviour.
_LOGISTIC_DENOM: Final = 1.0


def _sigmoid(value: int) -> int:
    """Apply the logistic sigmoid to a 32-bit signed register value.

    Operates in float64 (Python's native arithmetic) so the reference
    matches a numpy ``1/(1+exp(-x))`` to within 1 ULP; the result is
    rounded (banker's rounding via :func:`round`) back to int for
    storage in the i32 register file.
    """
    fx = float(value)
    result = 1.0 / (1.0 + math.exp(-fx))
    return round(result)


@dataclass(frozen=True, slots=True)
class InterpreterState:
    """Snapshot of the post-run register file."""

    registers: tuple[int, ...]


def _check_register(slot: int, what: str) -> None:
    """Reject register slots outside the 32-slot register file."""
    if not 0 <= slot < REGISTER_COUNT:
        raise ValueError(f"{what} register r{slot} out of range (0..31)")


def run(
    words: Sequence[int],
    *,
    memory: bytearray,
    registers: Sequence[int] | None = None,
) -> InterpreterState:
    """Execute a MicroGPU bytecode stream and return the final register file.

    ``memory`` is mutated in place by ``STORE_GLOBAL`` and read by
    ``LOAD_GLOBAL``. ``registers`` is the initial register file (one
    int per slot); if omitted, all 32 slots start at zero. The run
    loop terminates when ``RETURN`` is decoded or the program counter
    falls off the end of the stream.
    """
    regs = list(registers) if registers is not None else [0] * REGISTER_COUNT
    if len(regs) != REGISTER_COUNT:
        raise ValueError(
            f"registers must have {REGISTER_COUNT} entries, got {len(regs)}"
        )

    pc = 0
    while pc < len(words):
        instr = decode(words[pc])
        match instr.opcode:
            case OpCode.LOAD_GLOBAL:
                _check_register(instr.dst, "LOAD_GLOBAL dst")
                addr = instr.imm * BYTES_PER_WORD
                end = addr + BYTES_PER_WORD
                if end > len(memory):
                    raise IndexError(
                        f"LOAD_GLOBAL OOB: addr={addr} + {BYTES_PER_WORD} > {len(memory)}"
                    )
                regs[instr.dst] = int.from_bytes(
                    memory[addr:end], "little", signed=True
                )
            case OpCode.STORE_GLOBAL:
                _check_register(instr.src, "STORE_GLOBAL src")
                addr = instr.imm * BYTES_PER_WORD
                end = addr + BYTES_PER_WORD
                if end > len(memory):
                    raise IndexError(
                        f"STORE_GLOBAL OOB: addr={addr} + {BYTES_PER_WORD} > {len(memory)}"
                    )
                memory[addr:end] = (regs[instr.src] & 0xFFFFFFFF).to_bytes(
                    BYTES_PER_WORD, "little", signed=False
                )
            case OpCode.ADD:
                _check_register(instr.dst, "ADD dst")
                _check_register(instr.src, "ADD src")
                regs[instr.dst] = (regs[instr.dst] + regs[instr.src]) & 0xFFFFFFFF
            case OpCode.MUL:
                _check_register(instr.dst, "MUL dst")
                _check_register(instr.src, "MUL src")
                regs[instr.dst] = (regs[instr.dst] * regs[instr.src]) & 0xFFFFFFFF
            case OpCode.SIGMOID:
                _check_register(instr.dst, "SIGMOID dst")
                regs[instr.dst] = _sigmoid(regs[instr.dst])
            case OpCode.RETURN:
                break
            case OpCode.NOP:
                pass
        pc += 1

    return InterpreterState(registers=tuple(regs))


__all__ = (
    "BYTES_PER_WORD",
    "InterpreterState",
    "REGISTER_COUNT",
    "run",
)