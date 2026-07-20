"""Text assembler/disassembler tests for the MicroGPU SIMT bytecode ISA."""

from __future__ import annotations

import pytest

from backends.microgpu.asm import (
    MicrogpuOp,
    assemble,
    assemble_from_microgpu_dialect,
    disassemble,
    disassemble_to_microgpu_dialect,
)
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


# ---------------------------------------------------------------------------
# Dialect round-trip (Task 72)
# ---------------------------------------------------------------------------


DIALECT_CASES: list[tuple[MicrogpuOp, int, OpCode, int, int, int]] = [
    # (op, expected word, expected opcode, expected dst, expected src, expected imm)
    (
        MicrogpuOp(name="load_global", dst=2, imm=5),
        encode(OpCode.LOAD_GLOBAL, 2, 0, imm=5),
        OpCode.LOAD_GLOBAL,
        2,
        0,
        5,
    ),
    (
        MicrogpuOp(name="store_global", src=3, imm=6),
        encode(OpCode.STORE_GLOBAL, 0, 3, imm=6),
        OpCode.STORE_GLOBAL,
        0,
        3,
        6,
    ),
    (
        MicrogpuOp(name="add", dst=1, src=2),
        encode(OpCode.ADD, 1, 2, imm=0),
        OpCode.ADD,
        1,
        2,
        0,
    ),
    (
        MicrogpuOp(name="mul", dst=1, src=2),
        encode(OpCode.MUL, 1, 2, imm=0),
        OpCode.MUL,
        1,
        2,
        0,
    ),
    (
        MicrogpuOp(name="sigmoid", dst=4),
        encode(OpCode.SIGMOID, 4, 0, imm=0),
        OpCode.SIGMOID,
        4,
        0,
        0,
    ),
    (
        MicrogpuOp(name="return"),
        encode(OpCode.RETURN, 0, 0, imm=0),
        OpCode.RETURN,
        0,
        0,
        0,
    ),
]


@pytest.mark.parametrize(("op", "expected_word", "_opc", "_dst", "_src", "_imm"), DIALECT_CASES)
def test_assemble_from_microgpu_dialect_emits_expected_word(
    op: MicrogpuOp, expected_word: int, _opc: OpCode, _dst: int, _src: int, _imm: int
) -> None:
    """Given one dialect op record, the assembler emits the matching 16-bit word."""
    assert assemble_from_microgpu_dialect([op]) == [expected_word]


@pytest.mark.parametrize(("op", "_word", "expected_opc", "expected_dst", "expected_src", "expected_imm"), DIALECT_CASES)
def test_disassemble_to_microgpu_dialect_emits_expected_record(
    op: MicrogpuOp,
    _word: int,
    expected_opc: OpCode,
    expected_dst: int,
    expected_src: int,
    expected_imm: int,
) -> None:
    """Given a 16-bit word, the disassembler emits a record with the right field shape."""
    record = disassemble_to_microgpu_dialect([_word])[0]
    assert record.name == op.name
    assert record.dst == expected_dst
    assert record.src == expected_src
    assert record.imm == expected_imm
    assert decode(_word).opcode is expected_opc


def test_dialect_round_trip_covers_all_six_dialect_ops() -> None:
    """All six dialect ops survive a full dialect -> bytes -> dialect round-trip."""
    ops: list[MicrogpuOp] = [case[0] for case in DIALECT_CASES]
    words = assemble_from_microgpu_dialect(ops)
    rebuilt = disassemble_to_microgpu_dialect(words)
    assert rebuilt == ops
    assert assemble_from_microgpu_dialect(rebuilt) == words


def test_dialect_assemble_rejects_unknown_op_name() -> None:
    """Given an op name outside the dialect, the bridge raises."""
    op = MicrogpuOp(name="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown microgpu dialect op"):
        _ = assemble_from_microgpu_dialect([op])


def test_dialect_disassemble_rejects_nop_word() -> None:
    """Given a NOP word, the dialect bridge raises (no dialect op exists)."""
    nop_word = encode(OpCode.NOP, 0, 0, imm=0)
    with pytest.raises(ValueError, match="NOP has no MicroGPU dialect op"):
        _ = disassemble_to_microgpu_dialect([nop_word])


def test_dialect_round_trip_matches_text_assembler() -> None:
    """The dialect bridge and text assembler emit the same byte stream."""
    ops: list[MicrogpuOp] = [case[0] for case in DIALECT_CASES]
    text = disassemble(assemble_from_microgpu_dialect(ops))
    assert assemble(text) == assemble_from_microgpu_dialect(ops)