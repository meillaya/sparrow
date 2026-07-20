"""Race + OOB detection tests for the MicroGPU SIMT bytecode interpreter.

The interpreter layers two diagnostics on top of the run loop (Tasks
76 and 77): a per-register WAR/WAW hazard warning (non-fatal) and an
out-of-bounds :class:`MemoryOutOfBoundsError` (fatal). This file
exercises both surfaces.
"""

from __future__ import annotations

import warnings

import pytest

from backends.microgpu.asm import assemble
from backends.microgpu.interp import (
    BYTES_PER_WORD,
    MemoryOutOfBoundsError,
    RaceWarning,
    run,
)


def test_war_hazard_emits_race_warning() -> None:
    """A write to r0 after an earlier read of r0 emits a WAR hazard warning.

    Program:

        ADD  r1, r0     # reads r0
        ADD  r0, r1     # writes r0 (WAR)

    The interpreter must emit a ``RaceWarning`` for the second
    instruction but still execute the program.
    """
    program = (
        "ADD  r1, r0\n"
        "ADD  r0, r1\n"
        "RETURN\n"
    )
    memory = bytearray(BYTES_PER_WORD)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = run(assemble(program), memory=memory, registers=[3, 0] + [0] * 30)

    war = [w for w in caught if issubclass(w.category, RaceWarning)]
    assert war, "expected a RaceWarning for the WAR hazard"
    assert "WAR" in str(war[0].message)
    assert state.registers[0] == 6
    assert state.registers[1] == 3


def test_waw_hazard_emits_race_warning_with_intervening_read() -> None:
    """A write to r0 after another write + intervening read emits a WAW hazard.

    Program:

        LOAD_GLOBAL  r0, [0x0]   # writes r0 (no prior read)
        ADD  r1, r0              # reads r0 (intervening read), writes r1
        ADD  r0, r1              # writes r0 — WAW with intervening read

    The first LOAD writes r0 without any prior read of r0, so it is
    not a WAR. The third ADD's write to r0 has both a prior read AND
    a prior write of r0 — the WAW pattern fires.
    """
    program = (
        "LOAD_GLOBAL  r0, [0x0]\n"
        "ADD  r1, r0\n"
        "ADD  r0, r1\n"
        "RETURN\n"
    )
    memory = bytearray(BYTES_PER_WORD * 2)
    memory[:BYTES_PER_WORD] = (5 & 0xFFFFFFFF).to_bytes(BYTES_PER_WORD, "little", signed=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(assemble(program), memory=memory)

    waw = [
        w for w in caught
        if issubclass(w.category, RaceWarning) and "WAW" in str(w.message)
    ]
    assert waw, "expected a RaceWarning for the WAW hazard"


def test_plain_flow_dependency_does_not_emit_warning() -> None:
    """A plain RAW dep (``LOAD; ADD`` on the same slot) emits no race.

    Loading into r0 then immediately writing r0 via ADD/MUL/SIGMOID is
    the standard flow dependency — no intervening reader is at risk.
    The interpreter must NOT emit a ``RaceWarning`` for that pattern.
    """
    program = (
        "LOAD_GLOBAL  r0, [0x0]\n"
        "ADD  r0, r0\n"
        "STORE_GLOBAL r0, [0x1]\n"
        "RETURN\n"
    )
    memory = bytearray(BYTES_PER_WORD * 2)
    memory[:BYTES_PER_WORD] = (7 & 0xFFFFFFFF).to_bytes(BYTES_PER_WORD, "little", signed=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(assemble(program), memory=memory)

    race_warnings = [w for w in caught if issubclass(w.category, RaceWarning)]
    assert race_warnings == []


def test_detect_races_false_silences_warnings() -> None:
    """``detect_races=False`` disables the warning emitter."""
    program = (
        "ADD  r1, r0\n"
        "ADD  r0, r1\n"
        "RETURN\n"
    )
    memory = bytearray(BYTES_PER_WORD)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(assemble(program), memory=memory, registers=[3, 0] + [0] * 30, detect_races=False)

    assert not [w for w in caught if issubclass(w.category, RaceWarning)]

# ---------------------------------------------------------------------------
# Out-of-bounds detection (Task 77)
# ---------------------------------------------------------------------------


def test_load_global_raises_on_oob_address() -> None:
    """A LOAD at an address that overruns the buffer raises OOB."""
    program = "LOAD_GLOBAL  r0, [0xF]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 4)

    with pytest.raises(MemoryOutOfBoundsError) as exc:
        _ = run(assemble(program), memory=memory)

    assert exc.value.opcode == "LOAD_GLOBAL"
    assert exc.value.address == 0xF * BYTES_PER_WORD
    assert exc.value.memory_size == len(memory)


def test_store_global_raises_on_oob_address() -> None:
    """A STORE at an address that overruns the buffer raises OOB."""
    program = (
        "LOAD_GLOBAL  r0, [0x0]\n"
        "STORE_GLOBAL r0, [0xF]\n"
        "RETURN\n"
    )
    memory = bytearray(BYTES_PER_WORD * 4)

    with pytest.raises(MemoryOutOfBoundsError) as exc:
        _ = run(assemble(program), memory=memory)

    assert exc.value.opcode == "STORE_GLOBAL"
    assert exc.value.address == 0xF * BYTES_PER_WORD
    assert exc.value.memory_size == len(memory)


def test_oob_is_an_index_error_subclass() -> None:
    """``MemoryOutOfBoundsError`` is catchable as ``IndexError`` too."""
    program = "STORE_GLOBAL r0, [0xE]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 2)

    with pytest.raises(IndexError):
        _ = run(assemble(program), memory=memory)


def test_load_at_last_word_succeeds() -> None:
    """A LOAD at the highest valid word address returns the buffered value."""
    program = "LOAD_GLOBAL  r0, [0x3]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 4)
    memory[3 * BYTES_PER_WORD : 4 * BYTES_PER_WORD] = (99 & 0xFFFFFFFF).to_bytes(
        BYTES_PER_WORD, "little", signed=False
    )

    state = run(assemble(program), memory=memory)
    assert state.registers[0] == 99


def test_store_at_last_word_succeeds() -> None:
    """A STORE at the highest valid word address writes the buffered value."""
    program = (
        "LOAD_GLOBAL  r0, [0x0]\n"
        "STORE_GLOBAL r0, [0x3]\n"
        "RETURN\n"
    )
    memory = bytearray(BYTES_PER_WORD * 4)
    memory[:BYTES_PER_WORD] = (1234 & 0xFFFFFFFF).to_bytes(BYTES_PER_WORD, "little", signed=False)

    run(assemble(program), memory=memory)
    assert int.from_bytes(memory[3 * BYTES_PER_WORD : 4 * BYTES_PER_WORD], "little") == 1234
