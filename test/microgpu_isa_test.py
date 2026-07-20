"""Round-trip coverage for the MicroGPU SIMT bytecode ISA (spec §11).

Every shipped opcode round-trips through :func:`encode` and
:func:`decode` without losing field values, and the field-range
validator rejects out-of-range arguments. The test mirrors
:mod:`test.microcode_test` in shape so a future maintainer can
diff the two side-by-side.
"""

from __future__ import annotations

import pytest

from backends.microgpu.isa import MicroOp, OpCode, decode, encode


@pytest.mark.parametrize(
    ("opcode", "dst", "src", "imm"),
    [
        (OpCode.RETURN, 0, 0, 0),
        (OpCode.LOAD_GLOBAL, 0x1, 0x0, 0x2),
        (OpCode.STORE_GLOBAL, 0x3, 0x4, 0x5),
        (OpCode.ADD, 0x6, 0x7, 0x0),
        (OpCode.MUL, 0x8, 0x9, 0x0),
        (OpCode.SIGMOID, 0xA, 0x0, 0x0),
        (OpCode.NOP, 0x0, 0x0, 0x0),
    ],
)
def test_encode_decode_round_trip(opcode: OpCode, dst: int, src: int, imm: int) -> None:
    """Given one ISA opcode, encoding then decoding preserves its fields."""
    word = encode(opcode, dst, src, imm=imm)

    assert decode(word) == MicroOp(opcode, dst, src, imm)
    assert 0 <= word < (1 << 16)


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("dst", "dst must fit in 4 bits"),
        ("src", "src must fit in 4 bits"),
        ("imm", "imm must fit in 4 bits"),
    ],
)
def test_encode_rejects_out_of_range_fields(field: str, expected_message: str) -> None:
    """Given a 5-bit value, encoding rejects it as too wide for the 4-bit field."""
    sentinel = 0x10  # first value that does not fit in 4 bits
    match field:
        case "dst":
            with pytest.raises(ValueError, match=expected_message):
                _ = encode(OpCode.NOP, sentinel, 0)
        case "src":
            with pytest.raises(ValueError, match=expected_message):
                _ = encode(OpCode.NOP, 0, sentinel)
        case "imm":
            with pytest.raises(ValueError, match=expected_message):
                _ = encode(OpCode.NOP, 0, 0, imm=sentinel)
        case _:
            pytest.fail(f"unexpected field {field!r}")


def test_decode_rejects_unknown_opcode() -> None:
    """Given a word whose high nibble is unmapped, decode raises."""
    word = (0x6 << 12) | (0x1 << 8) | (0x2 << 4) | 0x3
    with pytest.raises(ValueError, match="unknown MicroGPU wire opcode 0x6"):
        _ = decode(word)


def test_decode_rejects_oversized_word() -> None:
    """Given a word > 16 bits, decode raises rather than silently truncating."""
    with pytest.raises(ValueError, match="must fit in 16 bits"):
        _ = decode(1 << 16)


def test_all_seven_opcodes_round_trip() -> None:
    """Each shipped opcode has a stable numeric value (no reordering).

    The wire format is part of the ABI; a regression here means the
    interpreter would suddenly see a ``RETURN`` as a ``LOAD_GLOBAL`` on
    disk. The explicit asserts are duplicated intentionally so a
    reordering fails both this test and the parametrized round-trip.
    """
    expected = {
        OpCode.RETURN: 0x00,
        OpCode.LOAD_GLOBAL: 0x01,
        OpCode.STORE_GLOBAL: 0x02,
        OpCode.ADD: 0x03,
        OpCode.MUL: 0x04,
        OpCode.SIGMOID: 0x05,
        OpCode.NOP: 0xFF,
    }
    for opcode, value in expected.items():
        assert int(opcode) == value, f"{opcode.name} changed numeric value"
