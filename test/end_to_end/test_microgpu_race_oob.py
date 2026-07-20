"""End-to-end race + OOB detection pytest for the MicroGPU interpreter (Phase 8, Task 82).

Confirms the two diagnostic surfaces the interpreter layers on top of the
run loop (Tasks 76 + 77) actually fire when a hand-written bytecode
program exhibits the corresponding hazard:

* **Race detection** (Task 76) — a write-after-read (WAR) or a
  write-after-write-with-intervening-read (WAW) on the same register
  slot emits a :class:`backends.microgpu.interp.RaceWarning` on
  ``stderr``. Detection is non-fatal; the offending program still
  runs to completion so the diagnostics surface without aborting the
  host process.

* **Out-of-bounds detection** (Task 77) — a ``LOAD_GLOBAL`` /
  ``STORE_GLOBAL`` whose immediate runs past the caller's buffer
  raises :class:`backends.microgpu.interp.MemoryOutOfBoundsError`.
  Detection is fatal: the interpreter aborts the launch on the
  violating instruction.

The companion unit-level tests in ``test/microgpu_race_oob_test.py``
(Task 76 / 77) cover the same surfaces with ``warnings.catch_warnings``.
This file is the Phase-8 e2e mirror that exercises the same hazards
through the standard Python :func:`warnings.catch_warnings` +
:func:`pytest.raises` machinery at the end-to-end test layer, so a
regression in either diagnostic is caught by the e2e gate alongside
the FusedMLP-on-microgpu pytest (Task 81).
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

# ---------------------------------------------------------------------------
# Race detection (Task 76)
# ---------------------------------------------------------------------------


def test_war_race_emits_race_warning_to_stderr() -> None:
    """A write-after-read on r0 emits a :class:`RaceWarning`.

    Program:

        ADD  r1, r0     # reads r0
        ADD  r0, r1     # writes r0 (WAR hazard)

    The first ADD puts r0 into ``reg_reads``; the second ADD writes
    r0, which the interpreter flags as a WAR hazard. The warning is
    non-fatal — the program still runs to completion (r0 = 6, r1 = 3
    with the seeded initial registers).
    """
    program = "ADD  r1, r0\nADD  r0, r1\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = run(assemble(program), memory=memory, registers=[3, 0] + [0] * 30)

    war = [w for w in caught if issubclass(w.category, RaceWarning) and "WAR" in str(w.message)]
    assert war, "expected a RaceWarning for the WAR hazard"
    assert state.registers[0] == 6
    assert state.registers[1] == 3


def test_waw_race_emits_race_warning_to_stderr() -> None:
    """A write-after-write with intervening read emits a :class:`RaceWarning`.

    Program:

        LOAD_GLOBAL  r0, [0x0]   # writes r0 (no prior read)
        ADD  r1, r0              # reads r0 (intervening read)
        ADD  r0, r1              # writes r0 — WAW with intervening read

    The first LOAD writes r0 with no prior reader (so no WAR); the
    third ADD's write to r0 has both a prior reader AND a prior writer
    — the WAW pattern fires.
    """
    program = "LOAD_GLOBAL  r0, [0x0]\nADD  r1, r0\nADD  r0, r1\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 2)
    memory[:BYTES_PER_WORD] = (5 & 0xFFFFFFFF).to_bytes(BYTES_PER_WORD, "little", signed=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(assemble(program), memory=memory)

    waw = [w for w in caught if issubclass(w.category, RaceWarning) and "WAW" in str(w.message)]
    assert waw, "expected a RaceWarning for the WAW hazard"


def test_plain_flow_dependency_emits_no_race_warning() -> None:
    """A standard read-after-write flow dependency emits NO race warning.

    Loading into r0 then immediately writing r0 via ADD/MUL/SIGMOID is
    the standard flow dependency — no intervening reader is at risk of
    observing an in-flight value. The interpreter must NOT emit a
    :class:`RaceWarning` for that pattern; only WAR/WAW fire.
    """
    program = "LOAD_GLOBAL  r0, [0x0]\nADD  r0, r0\nSTORE_GLOBAL r0, [0x1]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 2)
    memory[:BYTES_PER_WORD] = (7 & 0xFFFFFFFF).to_bytes(BYTES_PER_WORD, "little", signed=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(assemble(program), memory=memory)

    race_warnings = [w for w in caught if issubclass(w.category, RaceWarning)]
    assert race_warnings == []


def test_detect_races_false_silences_race_warnings() -> None:
    """``detect_races=False`` disables the WAR/WAW emitter."""
    program = "ADD  r1, r0\nADD  r0, r1\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(assemble(program), memory=memory, registers=[3, 0] + [0] * 30, detect_races=False)

    assert not [w for w in caught if issubclass(w.category, RaceWarning)]


# ---------------------------------------------------------------------------
# Out-of-bounds detection (Task 77)
# ---------------------------------------------------------------------------


def test_load_global_oob_raises_memory_out_of_bounds_error() -> None:
    """A LOAD whose immediate overruns the buffer raises :class:`MemoryOutOfBoundsError`.

    Wire-format imm field is 4 bits wide so the largest immediate is
    ``0xF`` (= 15 words = 60 bytes). The 16-byte (4-word) buffer can't
    satisfy a read at offset 60, so the interpreter must reject the
    instruction with :class:`MemoryOutOfBoundsError`.
    """
    program = "LOAD_GLOBAL  r0, [0xF]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 4)

    with pytest.raises(MemoryOutOfBoundsError) as exc:
        _ = run(assemble(program), memory=memory)

    assert exc.value.opcode == "LOAD_GLOBAL"
    assert exc.value.address == 0xF * BYTES_PER_WORD
    assert exc.value.memory_size == len(memory)


def test_store_global_oob_raises_memory_out_of_bounds_error() -> None:
    """A STORE whose immediate overruns the buffer raises :class:`MemoryOutOfBoundsError`."""
    program = "LOAD_GLOBAL  r0, [0x0]\nSTORE_GLOBAL r0, [0xF]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 4)

    with pytest.raises(MemoryOutOfBoundsError) as exc:
        _ = run(assemble(program), memory=memory)

    assert exc.value.opcode == "STORE_GLOBAL"
    assert exc.value.address == 0xF * BYTES_PER_WORD
    assert exc.value.memory_size == len(memory)


def test_oob_exception_is_an_index_error_subclass() -> None:
    """``MemoryOutOfBoundsError`` is also catchable as :class:`IndexError`."""
    program = "STORE_GLOBAL r0, [0xE]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 2)

    with pytest.raises(IndexError):
        _ = run(assemble(program), memory=memory)


def test_load_at_last_word_address_succeeds() -> None:
    """A LOAD at the highest valid word address returns the buffered value."""
    program = "LOAD_GLOBAL  r0, [0x3]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 4)
    memory[3 * BYTES_PER_WORD : 4 * BYTES_PER_WORD] = (99 & 0xFFFFFFFF).to_bytes(
        BYTES_PER_WORD, "little", signed=False
    )

    state = run(assemble(program), memory=memory)
    assert state.registers[0] == 99


def test_store_at_last_word_address_succeeds() -> None:
    """A STORE at the highest valid word address writes the buffered value."""
    program = "LOAD_GLOBAL  r0, [0x0]\nSTORE_GLOBAL r0, [0x3]\nRETURN\n"
    memory = bytearray(BYTES_PER_WORD * 4)
    memory[:BYTES_PER_WORD] = (1234 & 0xFFFFFFFF).to_bytes(BYTES_PER_WORD, "little", signed=False)

    run(assemble(program), memory=memory)
    assert int.from_bytes(memory[3 * BYTES_PER_WORD : 4 * BYTES_PER_WORD], "little") == 1234
