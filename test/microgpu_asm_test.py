"""Text assembler/disassembler tests for the MicroGPU SIMT bytecode ISA."""

from __future__ import annotations

import pytest

from backends.microgpu.asm import assemble, disassemble
from backends.microgpu.isa import OpCode, decode, encode


TEXT_CASES: list[tuple[str, int]] = [
    # (text, expected 16-bit word)
    ("LOAD_GLOBAL  r0, [0x1]", encode(OpCode.LOAD_GLOBAL, 0, 0, imm=0x1)),
    ("LOAD_GLOBAL  r1, [0x2]", encode(OpCode.LOAD_GLOBAL, 1, 0, imm=0x2)),
    ("STORE_GLOBAL r3, [0x4]", encode(OpCode.STORE_GLOBAL, 0, 3, imm=0x4)),
    ("ADD  r2, r3", encode(OpCode.ADD, 2, 3, imm=0)),
    ("MUL  r2, r3", encode(OpCode.MUL, 2, 3, imm=0)),
    ("SIGMOID r4", encode(OpCode.SIGMOID, 4, 0, imm=0)),
    ("RETURN", encode(OpCode.RETURN, 0, 0, imm=0)),
    ("NOP", encode(OpCode.NOP, 0, 0, imm=0)),
]


@pytest.mark.parametrize(("text", "expected"), TEXT_CASES)
def test_assemble_spec_operation(text: str, expected: int) -> None:
    """Given one textual operation, assembly produces its instruction word."""
    assert assemble(text) == [expected]


@pytest.mark.parametrize(("text", "_expected"), TEXT_CASES)
def test_disassemble_round_trip(text: str, _expected: int) -> None:
    """Given one instruction word, disassembly then reassembly is identity."""
    words = assemble(text)

    round_trip_text = disassemble(words)
    assert assemble(round_trip_text) == words


def test_assemble_disassemble_round_trip_for_full_program() -> None:
    """A multi-instruction program survives a full assemble -> disassemble cycle."""
    text = "\n".join(case[0] for case in TEXT_CASES)
    words = assemble(text)
    rebuilt_words = assemble(disassemble(words))
    assert rebuilt_words == words


def test_assemble_ignores_blank_lines_and_comments() -> None:
    """Given formatting-only lines, assembly emits only executable operations."""
    text = "\n# load x\nLOAD_GLOBAL  r0, [0x1]\n\n# multiply\nMUL  r0, r0\n"

    assert assemble(text) == [
        encode(OpCode.LOAD_GLOBAL, 0, 0, imm=0x1),
        encode(OpCode.MUL, 0, 0, imm=0),
    ]


def test_assemble_rejects_malformed_register() -> None:
    """Given a register slot > 15, assembly reports the line number."""
    with pytest.raises(ValueError, match=r"line 1.*register 'r16'"):
        _ = assemble("ADD  r16, r0")


def test_assemble_rejects_malformed_address() -> None:
    """Given an out-of-range address, assembly reports the line number."""
    with pytest.raises(ValueError, match=r"line 1.*address '\[0x100\]'"):
        _ = assemble("LOAD_GLOBAL  r0, [0x100]")


def test_assemble_rejects_unknown_opcode() -> None:
    """Given an unknown mnemonic, assembly reports the line number."""
    with pytest.raises(ValueError, match=r"line 1.*unknown opcode 'FOO'"):
        _ = assemble("FOO r0, r1")


def test_assemble_rejects_wrong_operand_count() -> None:
    """Given the wrong number of operands, assembly reports the line number."""
    with pytest.raises(ValueError, match=r"line 1.*ADD expects 2 operands"):
        _ = assemble("ADD  r0")


def test_assemble_rejects_operands_on_return() -> None:
    """Given operands on RETURN, assembly rejects the line."""
    with pytest.raises(ValueError, match=r"line 1.*RETURN takes no operands"):
        _ = assemble("RETURN r0")


def test_disassemble_emits_canonical_layout() -> None:
    """The disassembler emits the canonical operand layout per opcode."""
    text = disassemble(
        [
            encode(OpCode.LOAD_GLOBAL, 0, 0, imm=0x1),
            encode(OpCode.STORE_GLOBAL, 0, 3, imm=0x4),
            encode(OpCode.ADD, 2, 3, imm=0),
            encode(OpCode.MUL, 2, 3, imm=0),
            encode(OpCode.SIGMOID, 4, 0, imm=0),
            encode(OpCode.RETURN, 0, 0, imm=0),
            encode(OpCode.NOP, 0, 0, imm=0),
        ]
    )

    assert text.splitlines() == [
        "LOAD_GLOBAL  r0, [0x1]",
        "STORE_GLOBAL r3, [0x4]",
        "ADD  r2, r3",
        "MUL  r2, r3",
        "SIGMOID r4",
        "RETURN",
        "NOP",
    ]


def test_disassemble_rejects_unknown_wire_opcode() -> None:
    """Given a word with an unmapped wire opcode, disassemble raises."""
    bad_word = (0x6 << 12) | (0x1 << 8) | (0x2 << 4) | 0x3
    with pytest.raises(ValueError, match="unknown MicroGPU wire opcode"):
        _ = disassemble([bad_word])


def test_all_seven_opcodes_have_text_round_trip() -> None:
    """Every shipped opcode emits the expected decoded fields."""
    cases = [
        (encode(OpCode.LOAD_GLOBAL, 0, 0, imm=0x5), OpCode.LOAD_GLOBAL, 0, 0, 0x5),
        (encode(OpCode.STORE_GLOBAL, 0, 3, imm=0x6), OpCode.STORE_GLOBAL, 0, 3, 0x6),
        (encode(OpCode.ADD, 2, 3, imm=0), OpCode.ADD, 2, 3, 0),
        (encode(OpCode.MUL, 2, 3, imm=0), OpCode.MUL, 2, 3, 0),
        (encode(OpCode.SIGMOID, 4, 0, imm=0), OpCode.SIGMOID, 4, 0, 0),
        (encode(OpCode.RETURN, 0, 0, imm=0), OpCode.RETURN, 0, 0, 0),
        (encode(OpCode.NOP, 0, 0, imm=0), OpCode.NOP, 0, 0, 0),
    ]
    for word, opcode, dst, src, imm in cases:
        decoded = decode(word)
        assert decoded.opcode is opcode
        assert decoded.dst == dst
        assert decoded.src == src
        assert decoded.imm == imm