"""Race-detection tests for the MicroGPU SIMT bytecode interpreter (Task 76).

The interpreter layers a non-fatal :class:`RaceWarning` on top of the
run loop: it tracks per-register reads/writes and emits the warning
for WAR (write-after-read) and WAW (write-after-write with
intervening read) hazards. This file exercises that surface in
isolation; the out-of-bounds tests live alongside once Task 77 lands.
"""

from __future__ import annotations

import warnings

import pytest

from backends.microgpu.asm import assemble
from backends.microgpu.interp import (
    BYTES_PER_WORD,
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