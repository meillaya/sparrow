# compiler/lib/Dialect/MicroGPU/__init__.py
#
# `microgpu` xDSL dialect — Phase 8 / Task 73.
#
# The MicroGPU dialect is the high-level IR for the single-warp SIMT
# backend (spec §11). Each op lowers to one 32-bit MicroGPU instruction
# word (see :mod:`backends.microgpu.isa`); the dialect therefore stays
# narrow and carries no lowering logic of its own — codegen is the
# assembler's job (Task 72), and the runtime provider wraps the
# Python interpreter.
#
# Operations (exactly 6 — see Task 73 spec):
#   * ``microgpu.load_global``  — read a 32-bit word from ``mem[addr]``
#     into ``dst`` (i32 register).
#   * ``microgpu.store_global`` — write ``src`` (i32 register) to
#     ``mem[addr]``.
#   * ``microgpu.add``          — ``dst = lhs + rhs`` (i32).
#   * ``microgpu.mul``          — ``dst = lhs * rhs`` (i32).
#   * ``microgpu.sigmoid``      — ``dst = sigmoid(src)`` (i32).
#   * ``microgpu.return``       — terminate the warp.
#
# Pointer convention: the address operands are typed ``i32`` (the
# interpreter treats the immediate as a byte offset into the global
# memory buffer; the shipped wire format constrains the address to
# ``[0, 15]``). This is deliberately simpler than the astra dialect's
# ``!astra.ptr<#astra.memory_space<...>>`` because the microgpu
# single-warp model has one flat memory region (spec §11).

from __future__ import annotations

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException

# i32 is the canonical register + address width for microgpu.
I32: IntegerType = IntegerType(32)


def _check_i32(value, what: str) -> None:
    """Assert that an SSA value carries the canonical ``i32`` register type."""
    if not isinstance(value.type, IntegerType) or value.type.width.data != 32:
        raise VerifyException(f"{what} must be i32, got {value.type}")


# ---------------------------------------------------------------------------
# microgpu.load_global — mem[addr] -> dst
# ---------------------------------------------------------------------------


@irdl_op_definition
class LoadGlobalOp(IRDLOperation):
    """``microgpu.load_global`` — read a 32-bit word from ``mem[addr]`` into ``dst``.

    The ``addr`` operand is typed ``i32`` (the interpreter treats the
    value as a byte offset; the wire format constrains it to
    ``[0, 15]`` via the immediate field — see :mod:`backends.microgpu.isa`).
    The result ``dst`` is an ``i32`` register.
    """

    name = "microgpu.load_global"
    addr = operand_def(IntegerType)
    dst = result_def(IntegerType)

    assembly_format = "$addr attr-dict `:` `(` type($addr) `)` `->` type($dst)"

    def __init__(self, addr: object, dst_type: IntegerType = I32) -> None:
        super().__init__(operands=[addr], result_types=[dst_type])

    def verify_(self) -> None:
        _check_i32(self.addr, "microgpu.load_global operand 'addr'")
        _check_i32(self.dst, "microgpu.load_global result 'dst'")


# ---------------------------------------------------------------------------
# microgpu.store_global — src -> mem[addr]
# ---------------------------------------------------------------------------


@irdl_op_definition
class StoreGlobalOp(IRDLOperation):
    """``microgpu.store_global`` — write ``src`` (i32 register) to ``mem[addr]``.

    Mirror of :class:`LoadGlobalOp` for the back-direction: the source
    must be an ``i32`` register and the ``addr`` operand is typed
    ``i32`` (interpreter treats it as a byte offset).
    """

    name = "microgpu.store_global"
    src = operand_def(IntegerType)
    addr = operand_def(IntegerType)

    assembly_format = "$src `,` $addr attr-dict `:` `(` type($src) `,` type($addr) `)`"

    def __init__(self, src: object, addr: object) -> None:
        super().__init__(operands=[src, addr])

    def verify_(self) -> None:
        _check_i32(self.src, "microgpu.store_global operand 'src'")
        _check_i32(self.addr, "microgpu.store_global operand 'addr'")


# ---------------------------------------------------------------------------
# microgpu.add — dst = lhs + rhs
# ---------------------------------------------------------------------------


@irdl_op_definition
class AddOp(IRDLOperation):
    """``microgpu.add`` — ``dst = lhs + rhs`` (i32 register-register add)."""

    name = "microgpu.add"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    dst = result_def(IntegerType)

    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($dst)"
    )

    def __init__(self, lhs: object, rhs: object, dst_type: IntegerType = I32) -> None:
        super().__init__(operands=[lhs, rhs], result_types=[dst_type])

    def verify_(self) -> None:
        _check_i32(self.lhs, "microgpu.add operand 'lhs'")
        _check_i32(self.rhs, "microgpu.add operand 'rhs'")
        _check_i32(self.dst, "microgpu.add result 'dst'")


# ---------------------------------------------------------------------------
# microgpu.mul — dst = lhs * rhs
# ---------------------------------------------------------------------------


@irdl_op_definition
class MulOp(IRDLOperation):
    """``microgpu.mul`` — ``dst = lhs * rhs`` (i32 register-register mul)."""

    name = "microgpu.mul"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    dst = result_def(IntegerType)

    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($dst)"
    )

    def __init__(self, lhs: object, rhs: object, dst_type: IntegerType = I32) -> None:
        super().__init__(operands=[lhs, rhs], result_types=[dst_type])

    def verify_(self) -> None:
        _check_i32(self.lhs, "microgpu.mul operand 'lhs'")
        _check_i32(self.rhs, "microgpu.mul operand 'rhs'")
        _check_i32(self.dst, "microgpu.mul result 'dst'")


# ---------------------------------------------------------------------------
# microgpu.sigmoid — dst = sigmoid(src)
# ---------------------------------------------------------------------------


@irdl_op_definition
class SigmoidOp(IRDLOperation):
    """``microgpu.sigmoid`` — ``dst = sigmoid(src)`` (i32 in-place sigmoid)."""

    name = "microgpu.sigmoid"
    src = operand_def(IntegerType)
    dst = result_def(IntegerType)

    assembly_format = "$src attr-dict `:` `(` type($src) `)` `->` type($dst)"

    def __init__(self, src: object, dst_type: IntegerType = I32) -> None:
        super().__init__(operands=[src], result_types=[dst_type])

    def verify_(self) -> None:
        _check_i32(self.src, "microgpu.sigmoid operand 'src'")
        _check_i32(self.dst, "microgpu.sigmoid result 'dst'")


# ---------------------------------------------------------------------------
# microgpu.return — terminate the warp
# ---------------------------------------------------------------------------


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """``microgpu.return`` — terminate the warp.

    Has no operands or results; the assembly format prints the bare
    mnemonic. Following MLIR conventions, the op name is ``return``
    (the dialect prefix disambiguates it from ``func.return`` /
    ``scf.return`` when they appear in the same module).
    """

    name = "microgpu.return"

    assembly_format = "attr-dict"

    def __init__(self) -> None:
        super().__init__()


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


MicroGPU = Dialect(
    "microgpu",
    [
        LoadGlobalOp,
        StoreGlobalOp,
        AddOp,
        MulOp,
        SigmoidOp,
        ReturnOp,
    ],
    [],
)


__all__ = (
    "I32",
    "AddOp",
    "LoadGlobalOp",
    "MicroGPU",
    "MulOp",
    "ReturnOp",
    "SigmoidOp",
    "StoreGlobalOp",
)
