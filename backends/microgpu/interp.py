"""MicroGPU SIMT bytecode interpreter (spec §11).

The interpreter is a tiny reference implementation: a single in-order
fetch/decode/execute loop over a 32-slot register file, with the
backing store carried in a caller-provided ``bytearray``. It is
deliberately small — the goal is "reference correctness" for the
assembler and dialect, not high-throughput simulation. The Python
reference is also what the C++ :class:`forge_rt::MicroGpuProvider`
ultimately delegates to (via subprocess), so the test surface lives
here.

This module layers three diagnostic features on top of the run loop:

* **Race detection** (Task 76) — tracks per-register reads/writes and
  emits a :class:`RaceWarning` to ``stderr`` for WAR (write-after-read)
  and WAW (write-after-write with intervening read) hazards. Detection
  is best-effort and non-fatal; the interpreter still executes the
  offending program so the diagnostics surface without aborting the
  host process.
* **Out-of-bounds detection** (Task 77) — on ``LOAD_GLOBAL`` and
  ``STORE_GLOBAL``, asserts that ``addr + BYTES_PER_WORD <= memory.size()``
  and raises :class:`MemoryOutOfBoundsError` on violation. This catches
  the most common shape bug in hand-written microcode (an immediate
  field that runs off the end of the caller's buffer).
* **Dependency DAG** (Task 78) — :func:`bytecode_to_dag` walks the
  bytecode once and emits the (instruction, reg-deps, mem-deps) DAG
  the Phase 7 partitioner consumes. This is a separate helper so the
  interpreter's hot path stays branchless.

Reference: spec §11.
"""

from __future__ import annotations

import math
import struct
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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


class RaceWarning(UserWarning):
    """Emitted on stderr for WAR/WAW register hazards (non-fatal)."""


class MemoryOutOfBoundsError(IndexError):
    """Raised when a LOAD/STORE_GLOBAL immediate exceeds the caller's buffer."""

    def __init__(self, opcode: str, address: int, memory_size: int) -> None:
        self.opcode = opcode
        self.address = address
        self.memory_size = memory_size
        super().__init__(f"{opcode} OOB: addr={address} + {BYTES_PER_WORD} > {memory_size}")


@dataclass(frozen=True, slots=True)
class InterpreterState:
    """Snapshot of the post-run register file."""

    registers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DepNode:
    """One node in the bytecode dep DAG (Task 78).

    ``index`` is the bytecode index of the instruction, ``name`` is the
    textual mnemonic, ``reg_reads`` / ``reg_writes`` are the register
    slots the instruction touches, and ``mem_reads`` / ``mem_writes``
    are the memory-address offsets (in words) the instruction touches.
    Edges between nodes are derived by the caller from these sets.
    """

    index: int
    name: str
    reg_reads: tuple[int, ...] = ()
    reg_writes: tuple[int, ...] = ()
    mem_reads: tuple[int, ...] = ()
    mem_writes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DepDAG:
    """Bytecode dep DAG (Task 78).

    A flat list of :class:`DepNode`s with edges encoded as a
    ``(from_index, to_index)`` list. The Phase 7 partitioner reads the
    nodes to build the per-tensor readiness windows.
    """

    nodes: tuple[DepNode, ...]
    edges: tuple[tuple[int, int], ...] = field(default_factory=tuple)


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


def _check_register(slot: int, what: str) -> None:
    """Reject register slots outside the 32-slot register file."""
    if not 0 <= slot < REGISTER_COUNT:
        raise ValueError(f"{what} register r{slot} out of range (0..31)")


def _check_oob(opcode_name: str, address: int, memory_size: int) -> None:
    """Raise :class:`MemoryOutOfBoundsError` if the load/store exceeds the buffer."""
    if address + BYTES_PER_WORD > memory_size:
        raise MemoryOutOfBoundsError(opcode_name, address, memory_size)


def _check_reg_race(
    slot: int,
    pc: int,
    *,
    reg_reads: set[int],
    reg_writes_after_reads: set[int],
    name: str,
) -> None:
    """Emit a :class:`RaceWarning` if ``slot`` participates in a WAR or WAW hazard.

    WAR (write-after-read): ``slot`` was read earlier and is now being
    written. The writer overwrites the value before some downstream
    consumer observes the read result.

    WAW (write-after-write with intervening read): ``slot`` was
    written earlier AND read in between (the ``reg_writes_after_reads``
    set), and is now being written again. Two writers with a read in
    between means the second writer's value is observable only after
    any concurrent reader has retired.

    Both warnings are emitted independently when both hazards apply.
    A plain WAW with no intervening read (e.g. ``LOAD_GLOBAL r0;
    SIGMOID r0``) is the standard flow dependency and is NOT a race —
    the previous write has no downstream readers, so the new write is
    observably equivalent.
    """
    if slot in reg_reads:
        warnings.warn(
            f"WAR hazard at pc={pc} ({name}): r{slot} read earlier; "
            f"the write now overwrites the source value before all readers retire",
            RaceWarning,
            stacklevel=3,
        )
    if slot in reg_writes_after_reads:
        warnings.warn(
            f"WAW hazard at pc={pc} ({name}): r{slot} written earlier and read in between; "
            f"the second write loses ordering against the intervening reader",
            RaceWarning,
            stacklevel=3,
        )


def run(
    words: Sequence[int],
    *,
    memory: bytearray,
    registers: Sequence[int] | None = None,
    detect_races: bool = True,
) -> InterpreterState:
    """Execute a MicroGPU bytecode stream and return the final register file.

    ``memory`` is mutated in place by ``STORE_GLOBAL`` and read by
    ``LOAD_GLOBAL``. ``registers`` is the initial register file (one
    int per slot); if omitted, all 32 slots start at zero. The run
    loop terminates when ``RETURN`` is decoded or the program counter
    falls off the end of the stream.

    ``detect_races`` enables per-register WAR/WAW hazard warnings
    (Task 76). The warnings are non-fatal; the program still executes
    to completion so the diagnostics surface without aborting the
    host. Set ``detect_races=False`` to silence the warnings (the C++
    provider uses this when running trusted, scheduler-validated
    programs).
    """
    regs = list(registers) if registers is not None else [0] * REGISTER_COUNT
    if len(regs) != REGISTER_COUNT:
        raise ValueError(f"registers must have {REGISTER_COUNT} entries, got {len(regs)}")

    reg_reads: set[int] = set()
    reg_writes: set[int] = set()
    reg_writes_after_reads: set[int] = set()

    pc = 0
    while pc < len(words):
        instr = decode(words[pc])
        match instr.opcode:
            case OpCode.LOAD_GLOBAL:
                _check_register(instr.dst, "LOAD_GLOBAL dst")
                addr = instr.imm * BYTES_PER_WORD
                _check_oob("LOAD_GLOBAL", addr, len(memory))
                if detect_races:
                    _check_reg_race(
                        instr.dst,
                        pc,
                        reg_reads=reg_reads,
                        reg_writes_after_reads=reg_writes_after_reads,
                        name="LOAD_GLOBAL",
                    )
                regs[instr.dst] = int.from_bytes(
                    memory[addr : addr + BYTES_PER_WORD], "little", signed=True
                )
                if detect_races:
                    reg_writes.add(instr.dst)
            case OpCode.STORE_GLOBAL:
                _check_register(instr.src, "STORE_GLOBAL src")
                addr = instr.imm * BYTES_PER_WORD
                _check_oob("STORE_GLOBAL", addr, len(memory))
                if detect_races:
                    reg_reads.add(instr.src)
                    if instr.src in reg_writes:
                        reg_writes_after_reads.add(instr.src)
                memory[addr : addr + BYTES_PER_WORD] = (regs[instr.src] & 0xFFFFFFFF).to_bytes(
                    BYTES_PER_WORD, "little", signed=False
                )
            case OpCode.ADD:
                _check_register(instr.dst, "ADD dst")
                _check_register(instr.src, "ADD src")
                if detect_races:
                    _check_reg_race(
                        instr.dst,
                        pc,
                        reg_reads=reg_reads,
                        reg_writes_after_reads=reg_writes_after_reads,
                        name="ADD",
                    )
                    reg_reads.add(instr.src)
                    if instr.src in reg_writes:
                        reg_writes_after_reads.add(instr.src)
                regs[instr.dst] = (regs[instr.dst] + regs[instr.src]) & 0xFFFFFFFF
                if detect_races:
                    reg_writes.add(instr.dst)
            case OpCode.MUL:
                _check_register(instr.dst, "MUL dst")
                _check_register(instr.src, "MUL src")
                if detect_races:
                    _check_reg_race(
                        instr.dst,
                        pc,
                        reg_reads=reg_reads,
                        reg_writes_after_reads=reg_writes_after_reads,
                        name="MUL",
                    )
                    reg_reads.add(instr.src)
                    if instr.src in reg_writes:
                        reg_writes_after_reads.add(instr.src)
                regs[instr.dst] = (regs[instr.dst] * regs[instr.src]) & 0xFFFFFFFF
                if detect_races:
                    reg_writes.add(instr.dst)
            case OpCode.FADD | OpCode.FMUL:
                _check_register(instr.dst, f"{instr.opcode.name} dst")
                _check_register(instr.src, f"{instr.opcode.name} src")
                left = struct.unpack("<f", struct.pack("<I", regs[instr.dst] & 0xFFFFFFFF))[0]
                right = struct.unpack("<f", struct.pack("<I", regs[instr.src] & 0xFFFFFFFF))[0]
                result = left + right if instr.opcode is OpCode.FADD else left * right
                regs[instr.dst] = struct.unpack("<I", struct.pack("<f", result))[0]
            case OpCode.FRELU:
                _check_register(instr.dst, "FRELU dst")
                value = struct.unpack("<f", struct.pack("<I", regs[instr.dst] & 0xFFFFFFFF))[0]
                regs[instr.dst] = struct.unpack("<I", struct.pack("<f", max(0.0, value)))[0]
            case OpCode.MATMUL:
                base = instr.imm * BYTES_PER_WORD
                _check_oob("MATMUL", base + 47 * BYTES_PER_WORD, len(memory))
                values = []
                for index in range(48):
                    offset = base + index * BYTES_PER_WORD
                    values.append(struct.unpack("<f", memory[offset : offset + BYTES_PER_WORD])[0])
                lhs = values[0:16]
                rhs = values[16:32]
                product = [
                    sum(lhs[row * 4 + inner] * rhs[inner * 4 + col] for inner in range(4))
                    for row in range(4)
                    for col in range(4)
                ]
                for index, value in enumerate(product):
                    offset = base + (32 + index) * BYTES_PER_WORD
                    memory[offset : offset + BYTES_PER_WORD] = struct.pack("<f", value)
            case OpCode.SIGMOID:
                _check_register(instr.dst, "SIGMOID dst")
                if detect_races:
                    _check_reg_race(
                        instr.dst,
                        pc,
                        reg_reads=reg_reads,
                        reg_writes_after_reads=reg_writes_after_reads,
                        name="SIGMOID",
                    )
                regs[instr.dst] = _sigmoid(regs[instr.dst])
                if detect_races:
                    reg_writes.add(instr.dst)
            case OpCode.RETURN:
                break
            case OpCode.NOP:
                pass
        pc += 1

    return InterpreterState(registers=tuple(regs))


def _instruction_deps(word: int, index: int) -> DepNode:
    """Build the per-instruction dep record (Task 78)."""
    instr = decode(word)
    match instr.opcode:
        case OpCode.LOAD_GLOBAL:
            return DepNode(
                index=index,
                name="LOAD_GLOBAL",
                reg_writes=(instr.dst,),
                mem_reads=(instr.imm,),
            )
        case OpCode.STORE_GLOBAL:
            return DepNode(
                index=index,
                name="STORE_GLOBAL",
                reg_reads=(instr.src,),
                mem_writes=(instr.imm,),
            )
        case OpCode.ADD | OpCode.MUL:
            return DepNode(
                index=index,
                name=instr.opcode.name,
                reg_reads=(instr.src,),
                reg_writes=(instr.dst,),
            )
        case OpCode.SIGMOID:
            return DepNode(
                index=index,
                name="SIGMOID",
                reg_reads=(instr.dst,),
                reg_writes=(instr.dst,),
            )
        case OpCode.FADD | OpCode.FMUL:
            return DepNode(
                index=index,
                name=instr.opcode.name,
                reg_reads=(instr.dst, instr.src),
                reg_writes=(instr.dst,),
            )
        case OpCode.FRELU:
            return DepNode(
                index=index,
                name="FRELU",
                reg_reads=(instr.dst,),
                reg_writes=(instr.dst,),
            )
        case OpCode.MATMUL:
            return DepNode(
                index=index,
                name="MATMUL",
                mem_reads=tuple(range(instr.imm, instr.imm + 32)),
                mem_writes=tuple(range(instr.imm + 32, instr.imm + 48)),
            )
        case OpCode.RETURN:
            return DepNode(index=index, name="RETURN")
        case OpCode.NOP:
            return DepNode(index=index, name="NOP")


def _edges_from_nodes(nodes: Iterable[DepNode]) -> tuple[tuple[int, int], ...]:
    """Compute the cross-node dep edges for a list of :class:`DepNode`s.

    Two nodes share an edge if one writes a register/memory slot the
    other reads, or both write the same slot (WAW dep). Edges are
    added in source order so the Phase 7 partitioner can iterate
    deterministically.
    """
    materialised = list(nodes)
    reg_last_write: dict[int, int] = {}
    mem_last_write: dict[int, int] = {}
    reg_last_read: dict[int, int] = {}
    mem_last_read: dict[int, int] = {}
    edges: list[tuple[int, int]] = []

    for node in materialised:
        for slot in node.reg_reads:
            if slot in reg_last_write:
                edges.append((reg_last_write[slot], node.index))
            reg_last_read[slot] = node.index
        for slot in node.reg_writes:
            if slot in reg_last_write:
                edges.append((reg_last_write[slot], node.index))
            if slot in reg_last_read:
                edges.append((reg_last_read[slot], node.index))
            reg_last_write[slot] = node.index
        for slot in node.mem_reads:
            if slot in mem_last_write:
                edges.append((mem_last_write[slot], node.index))
            mem_last_read[slot] = node.index
        for slot in node.mem_writes:
            if slot in mem_last_write:
                edges.append((mem_last_write[slot], node.index))
            if slot in mem_last_read:
                edges.append((mem_last_read[slot], node.index))
            mem_last_write[slot] = node.index

    return tuple(edges)


def bytecode_to_dag(words: Sequence[int]) -> DepDAG:
    """Convert a bytecode stream into a (nodes, edges) dep DAG (Task 78).

    The returned :class:`DepDAG` is the canonical input to the Phase 7
    partitioner. Nodes are returned in program order; edges are
    returned in topological-source order. The structure is purely a
    static analysis — no run-time state is captured.
    """
    nodes = tuple(_instruction_deps(word, index) for index, word in enumerate(words))
    edges = _edges_from_nodes(nodes)
    return DepDAG(nodes=nodes, edges=edges)


__all__ = (
    "BYTES_PER_WORD",
    "REGISTER_COUNT",
    "DepDAG",
    "DepNode",
    "InterpreterState",
    "MemoryOutOfBoundsError",
    "RaceWarning",
    "bytecode_to_dag",
    "run",
)
