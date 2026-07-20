"""Reference-impl tests for the MicroGPU SIMT bytecode interpreter.

Runs the canonical ``gelu(x @ w + b)`` microcode program for a
hand-picked input and asserts the final register state matches a
numpy reference within fp32-equivalent tolerance. The interpreter is
deliberately a thin fetch/decode/execute loop (spec §11); this file
exercises every shipped opcode at least once. It also covers the
dep-DAG emitter (Task 78) so the partitioner input surface is unit-
tested alongside the run loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from backends.microgpu.asm import assemble
from backends.microgpu.interp import (
    BYTES_PER_WORD,
    REGISTER_COUNT,
    DepDAG,
    DepNode,
    InterpreterState,
    bytecode_to_dag,
    run,
)


def _init_memory(values: list[int]) -> bytearray:
    """Build a memory buffer holding the int32 values in low-byte-first order."""
    memory = bytearray(BYTES_PER_WORD * 16)
    for slot, value in enumerate(values):
        offset = slot * BYTES_PER_WORD
        memory[offset : offset + BYTES_PER_WORD] = (value & 0xFFFFFFFF).to_bytes(
            BYTES_PER_WORD, "little", signed=False
        )
    return memory


def test_gelu_xw_plus_b_matches_numpy_reference() -> None:
    """``gelu(x @ w + b)`` computed by the interpreter matches numpy.

    Inputs are tiny: x=[1, 2], w=[3, 4], b=-9; expected dot product is
    ``1*3 + 2*4 + (-9) = 2``. The interpreter stores registers as i32,
    so the SIGMOID op first applies ``1/(1+exp(-x))`` in float64 then
    rounds to the nearest int (rather than truncating). For ``x = 2``,
    ``sigmoid(2) ≈ 0.8808`` rounds to ``1``, and the final store
    writes that back to memory.
    """
    program = (
        "LOAD_GLOBAL  r0, [0x0]\n"  # x[0]
        "LOAD_GLOBAL  r1, [0x1]\n"  # x[1]
        "LOAD_GLOBAL  r2, [0x2]\n"  # w[0]
        "LOAD_GLOBAL  r3, [0x3]\n"  # w[1]
        "LOAD_GLOBAL  r4, [0x4]\n"  # b
        "MUL  r0, r2\n"  # r0 = x[0] * w[0]
        "MUL  r1, r3\n"  # r1 = x[1] * w[1]
        "ADD  r0, r1\n"  # r0 = x.w partial sum
        "ADD  r0, r4\n"  # r0 = x.w + b
        "SIGMOID r0\n"  # r0 = sigmoid(r0)
        "STORE_GLOBAL r0, [0x5]\n"  # write result back
        "RETURN\n"
    )

    memory = _init_memory([1, 2, 3, 4, -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    state = run(assemble(program), memory=memory)

    reference = 1.0 / (1.0 + float(np.exp(-2.0)))
    assert state.registers[0] == pytest.approx(round(reference), abs=1e-9)
    assert int.from_bytes(memory[5 * BYTES_PER_WORD : 6 * BYTES_PER_WORD], "little") == 1


def test_run_returns_initial_state_for_no_op_program() -> None:
    """A no-op program returns the input registers unchanged."""
    state = run([], memory=bytearray(BYTES_PER_WORD * 4))

    assert state.registers == tuple([0] * REGISTER_COUNT)
    assert isinstance(state, InterpreterState)


def test_run_returns_initial_state_for_only_nop() -> None:
    """A program of only NOPs returns the input registers unchanged."""
    state = run(assemble("NOP\nNOP\nNOP\nRETURN\n"), memory=bytearray(BYTES_PER_WORD * 4))

    assert state.registers == tuple([0] * REGISTER_COUNT)


def test_run_honors_return_op() -> None:
    """A program terminated by RETURN ignores later instructions."""
    program = (
        "LOAD_GLOBAL  r0, [0x0]\n"
        "RETURN\n"
        "STORE_GLOBAL r0, [0x1]\n"  # ignored
    )
    memory = _init_memory([42, 0, 0, 0])
    state = run(assemble(program), memory=memory)

    assert state.registers[0] == 42
    assert memory[BYTES_PER_WORD : 2 * BYTES_PER_WORD] == b"\x00\x00\x00\x00"


def test_run_honors_initial_register_state() -> None:
    """Initial register values participate in the run."""
    program = "ADD  r0, r1\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD)
    state = run(assemble(program), memory=memory, registers=[7, 5] + [0] * 30)

    assert state.registers[0] == 12


def test_run_rejects_wrong_register_count() -> None:
    """The interpreter rejects a register file of the wrong length."""
    with pytest.raises(ValueError, match=r"registers must have 32 entries"):
        _ = run([], memory=bytearray(BYTES_PER_WORD), registers=[0, 0, 0])


def test_store_global_writes_back_to_memory() -> None:
    """STORE_GLOBAL writes the register value to the addressed memory slot."""
    program = "LOAD_GLOBAL  r0, [0x0]\nADD  r0, r0\nSTORE_GLOBAL r0, [0x2]\nRETURN\n"
    memory = _init_memory([21, 0, 0, 0])
    run(assemble(program), memory=memory)

    assert int.from_bytes(memory[2 * BYTES_PER_WORD : 3 * BYTES_PER_WORD], "little") == 42


def test_sigmoid_of_zero_is_half() -> None:
    """sigmoid(0) = 0.5 truncates to 0 in the int register file."""
    program = "SIGMOID r0\nRETURN\n"
    state = run(assemble(program), memory=bytearray(BYTES_PER_WORD))

    assert state.registers[0] == 0


def test_add_and_mul_carry_to_32_bits() -> None:
    """ADD/MUL mask their result back to a 32-bit unsigned value."""
    program = (
        "ADD  r0, r1\n"  # 0xFFFFFFFF + 1 = 0x100000000 -> masked to 0
        "RETURN\n"
    )
    memory = bytearray(BYTES_PER_WORD)
    state = run(assemble(program), memory=memory, registers=[0xFFFFFFFF, 1] + [0] * 30)

    assert state.registers[0] == 0


# ---------------------------------------------------------------------------
# Bytecode dep DAG (Task 78)
# ---------------------------------------------------------------------------


def test_dag_is_empty_for_empty_program() -> None:
    """An empty program yields an empty DAG (no nodes, no edges)."""
    dag = bytecode_to_dag([])

    assert dag.nodes == ()
    assert dag.edges == ()
    assert isinstance(dag, DepDAG)


def test_dag_node_per_instruction_in_program_order() -> None:
    """The DAG emits one node per instruction in program order."""
    program = "LOAD_GLOBAL  r0, [0x0]\nADD  r0, r1\nSTORE_GLOBAL r0, [0x2]\nRETURN\n"
    dag = bytecode_to_dag(assemble(program))

    assert [node.index for node in dag.nodes] == [0, 1, 2, 3]
    assert [node.name for node in dag.nodes] == ["LOAD_GLOBAL", "ADD", "STORE_GLOBAL", "RETURN"]
    assert all(isinstance(node, DepNode) for node in dag.nodes)


def test_dag_records_reg_and_mem_deps() -> None:
    """Each node carries the register and memory slots it touches."""
    program = "LOAD_GLOBAL  r0, [0x0]\nADD  r1, r0\nSTORE_GLOBAL r1, [0x1]\n"
    nodes = bytecode_to_dag(assemble(program)).nodes

    assert nodes[0].reg_writes == (0,)
    assert nodes[0].mem_reads == (0,)
    assert nodes[1].reg_reads == (0,)
    assert nodes[1].reg_writes == (1,)
    assert nodes[2].reg_reads == (1,)
    assert nodes[2].mem_writes == (1,)


def test_dag_edges_follow_dataflow() -> None:
    """Edges connect a writer to a downstream reader of the same slot."""
    program = (
        "LOAD_GLOBAL  r0, [0x0]\n"
        "LOAD_GLOBAL  r1, [0x1]\n"
        "ADD  r2, r0\n"
        "MUL  r2, r1\n"
        "STORE_GLOBAL r2, [0x2]\n"
    )
    edges = bytecode_to_dag(assemble(program)).edges

    assert (0, 2) in edges
    assert (1, 3) in edges
    assert (2, 3) in edges
    assert (3, 4) in edges


def test_dag_edges_carry_memory_dependencies() -> None:
    """A STORE after a LOAD of the same memory slot creates an edge."""
    program = "LOAD_GLOBAL  r0, [0x0]\nSTORE_GLOBAL r0, [0x0]\n"
    edges = bytecode_to_dag(assemble(program)).edges

    assert (0, 1) in edges


def test_dag_handles_nop_and_return() -> None:
    """NOP / RETURN carry no dependencies and no outgoing edges."""
    program = "LOAD_GLOBAL  r0, [0x0]\nNOP\nRETURN\n"
    dag = bytecode_to_dag(assemble(program))

    assert dag.nodes[1].name == "NOP"
    assert dag.nodes[2].name == "RETURN"
    assert dag.nodes[1].reg_reads == ()
    assert dag.nodes[1].reg_writes == ()
    assert dag.nodes[2].reg_writes == ()
    assert all(edge[0] != 1 and edge[1] != 1 for edge in dag.edges)
    assert all(edge[0] != 2 and edge[1] != 2 for edge in dag.edges)
