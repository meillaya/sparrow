# compiler/lib/Dialect/Astra/__init__.py
#
# `astra` xDSL dialect — Wave 9 / Tasks 40-41.
#
# This is the Phase 4 (Astra) IR surface: the high-level `compute.matmul`
# command and the four `microcode.*` primitives that the 16x16 systolic
# matrix array + DMA engine + vector ALU execute (spec §9). The dialect
# carries the metadata the cycle-accurate simulator (Tasks 59-68) and the
# codegen pass (Task 48) need, but performs **no** lowering itself: the
# Tensor → Astra conversion lives in
# ``compiler/lib/Conversion/TensorToAstra/__init__.py`` (Task 42) and the
# dialect-to-microcode-bytes codegen lives in Task 48.
#
# Attributes:
#   * ``#astra.memory_space<dram | sram0 | sram1 | scratch>``
#     — parametric ``!astra.ptr<#astra.memory_space<...>>`` memory region.
#     DRAM is the host-facing off-chip memory; SRAM0 / SRAM1 are the two
#     halves of the 512 KB scratchpad; SCRATCH is the on-tile scratch
#     region used by the vector ALU. This is intentionally narrower than
#     the runtime ``forge_rt::MemorySpace`` (Host / Device0 / Device1) —
#     that enum is a multi-device runtime concept, while the astra
#     dialect models a single chip's address spaces (spec §9).
#
# Operations (exactly 5 — see Task 40 spec):
#   * ``astra.compute.matmul``        — high-level matmul command (M, N, K).
#   * ``astra.microcode.dma_load``    — DRAM -> SRAM copy.
#   * ``astra.microcode.dma_store``   — SRAM -> DRAM copy.
#   * ``astra.microcode.matmul``      — systolic matmul on sram tiles.
#   * ``astra.microcode.vector_gelu`` — elementwise GELU on a sram tile.
#
# Pointer convention: every pointer operand carries a
# ``!astra.ptr<#astra.memory_space<...>>`` type so the verifier can
# distinguish DRAM vs SRAM pointers statically — DRAM ptrs are
# ``!astra.ptr<#astra.memory_space<dram>>``, SRAM ptrs are one of
# ``<sram0>``, ``<sram1>``, ``<scratch>``. This mirrors the Wave 0
# ``forge.schedule`` style of carrying constraint metadata in the type
# system, and lets later pattern-matchers (Task 48 codegen) reason about
# memory spaces without parsing attrs.

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import IntAttr, IntegerType
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

# ---------------------------------------------------------------------------
# MemorySpace enum + attribute (Task 41 — folded into Task 40)
# ---------------------------------------------------------------------------


class MemorySpace(StrEnum):
    """Address-space regions for the astra chip (spec §9).

    * ``dram``   — host-facing off-chip DRAM.
    * ``sram0``  — first half of the 512 KB scratchpad.
    * ``sram1``  — second half of the 512 KB scratchpad.
    * ``scratch``— on-tile scratch region used by the vector ALU.

    This is intentionally narrower than the runtime
    :class:`forge_rt::MemorySpace` (Host / Device0 / Device1): the
    runtime enum is a multi-device concept, while this dialect models
    a single chip's address spaces.
    """

    DRAM = "dram"
    SRAM0 = "sram0"
    SRAM1 = "sram1"
    SCRATCH = "scratch"


@irdl_attr_definition
class MemorySpaceAttr(EnumAttribute[MemorySpace]):
    """``#astra.memory_space<dram | sram0 | sram1 | scratch>``.

    Mirrors the Wave 0 ``#forge.activation<...>`` style: the base
    ``EnumAttribute`` does not print angle brackets, so we override
    ``print_parameter`` / ``parse_parameter`` to wrap the enum literal.
    """

    name = "astra.memory_space"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_identifier_or_string_literal(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> MemorySpace:
        with parser.in_angle_brackets():
            ident = parser.parse_identifier()
            try:
                return MemorySpace(ident)
            except ValueError as exc:
                valid = ", ".join(repr(m.value) for m in MemorySpace)
                parser.raise_error(
                    f"unknown astra.memory_space {ident!r}; expected one of: {valid}"
                )
                # raise_error raises; the line below is unreachable but keeps
                # the type checker happy.
                raise VerifyException(  # pragma: no cover
                    f"unknown astra.memory_space {ident!r}"
                ) from exc


# ---------------------------------------------------------------------------
# PtrType — typed pointer carrying a MemorySpace
# ---------------------------------------------------------------------------


@irdl_attr_definition
class PtrType(ParametrizedAttribute, TypeAttribute):
    """``!astra.ptr<#astra.memory_space<dram|sram0|sram1|scratch>>``.

    All astra pointer operands (``dram_ptr``, ``sram_ptr``, ``sram_a``,
    ``sram_b``, ``sram_acc``, optional ``bias_ptr``) are typed by this
    parametric attribute, so the verifier can statically reject
    ``dram_ptr`` operands where an SRAM-typed operand is expected (and
    vice versa) before the runtime ever sees a bad address. The payload
    is a 64-bit address (mapped to i64 by the conversion pass — see Task
    42); the dialect itself stays agnostic about the address width.
    """

    name = "astra.ptr"
    space: MemorySpaceAttr

    def __init__(self, space: MemorySpaceAttr | MemorySpace | str) -> None:
        if isinstance(space, (MemorySpace, str)):
            space = MemorySpaceAttr(space if isinstance(space, MemorySpace) else MemorySpace(space))
        super().__init__(space)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            attr = parser.parse_attribute()
            if not isinstance(attr, MemorySpaceAttr):
                parser.raise_error(
                    f"expected #astra.memory_space inside !astra.ptr<...>, got {type(attr).__name__}"
                )
            return (attr,)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.space)


# Convenience helpers for tests and conversion pass — keep call-sites
# readable (`PtrType.dram()` rather than `PtrType(MemorySpaceAttr(MemorySpace.DRAM))`).
def _make_ptr(space: MemorySpace) -> PtrType:
    return PtrType(MemorySpaceAttr(space))


# Re-exported aliases on the type for symmetry with the enum.
for _space in MemorySpace:
    setattr(PtrType, _space.value.upper(), _make_ptr(_space))


# ---------------------------------------------------------------------------
# Helpers shared between ops
# ---------------------------------------------------------------------------


def _parse_pos_int_attr(parser: AttrParser, what: str) -> IntAttr:
    """Parse a non-negative integer literal and wrap it in ``IntAttr``.

    The integer literals on ``M``, ``N``, ``K``, and ``bytes`` are
    syntactic ``int64``'s — the verifier rejects ``<= 0`` (for dims)
    or ``< 0`` (for byte counts) so downstream passes can assume
    ``value >= 1`` (dims) or ``value >= 0`` (bytes).
    """
    val = parser.parse_optional_integer(allow_boolean=False, allow_negative=False)
    if val is None:
        parser.raise_error(f"expected non-negative integer for {what}")
    return IntAttr(val)  # type: ignore[arg-type]  (raise_error never returns)


def _parse_dims_attr_dict(parser: AttrParser) -> dict[str, Attribute]:
    """Parse ``{m = N, n = N, k = N}`` for the microcode matmul attrs.

    The assembly format emits the three dims via the attr-dict syntax
    (``m``, ``n``, ``k``) so users can override them per-op without
    changing the grammar. All three must parse to ``IntAttr``; the
    verifier downstream rejects non-positive values.
    """
    parsed: dict[str, Attribute] = {}
    for key in ("m", "n", "k"):
        attr = parser.parse_attribute()
        if not isinstance(attr, IntAttr):
            parser.raise_error(f"astra microcode matmul '{key}' must be an integer attr")
        parsed[key] = attr
    return parsed


def _check_ptr(value: SSAValue, what: str) -> PtrType:
    """Assert that ``value`` carries an ``!astra.ptr<...>`` and return its type."""
    if not isinstance(value.type, PtrType):
        raise VerifyException(f"{what} must be !astra.ptr, got {value.type}")
    return value.type


def _check_dram(value: SSAValue, what: str) -> None:
    """Assert that ``value`` carries an ``i64`` (the canonical dram_ptr width)."""
    if not isinstance(value.type, IntegerType) or value.type.width.data != 64:
        raise VerifyException(f"{what} must be i64, got {value.type}")


def _check_positive_int(attr: IntAttr, name: str) -> None:
    """Reject non-positive ``IntAttr`` values (used by ``verify_`` hooks)."""
    if attr.data < 1:
        raise VerifyException(f"astra {name} must be positive, got {attr.data}")


def _check_nonneg_int(attr: IntAttr, name: str) -> None:
    """Reject negative ``IntAttr`` values (bytes can be zero)."""
    if attr.data < 0:
        raise VerifyException(f"astra {name} must be non-negative, got {attr.data}")


# ---------------------------------------------------------------------------
# astra.compute.matmul — high-level command-processor matmul
# ---------------------------------------------------------------------------


@irdl_op_definition
class ComputeMatmulOp(IRDLOperation):
    """``astra.compute.matmul`` — high-level matmul command.

    A structural marker carrying the M / N / K of a single systolic-array
    invocation. The Phase 4 codegen pass (Task 48) wraps one of these in
    the per-tile DMA + microcode sequence; the functional / cycle-accurate
    simulators (Tasks 59-68) consume the M/N/K directly to size the
    matrix-array utilization counters.

    The op has no operands / results — it is metadata-only, much like
    ``forge.schedule.tiled_gemm`` (Wave 0 / Task 2). The conversion pass
    ``TensorToAstra`` (Task 42) emits one of these per tile when it
    lowers a ``forge.tensor.gemm`` into the astra stream.
    """

    name = "astra.compute.matmul"
    m = prop_def(IntAttr)
    n = prop_def(IntAttr)
    k = prop_def(IntAttr)

    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = "attr-dict"

    def __init__(self, m: int | IntAttr, n: int | IntAttr, k: int | IntAttr) -> None:
        super().__init__(
            properties={
                "m": IntAttr(m) if isinstance(m, int) else m,
                "n": IntAttr(n) if isinstance(n, int) else n,
                "k": IntAttr(k) if isinstance(k, int) else k,
            }
        )

    def verify_(self) -> None:
        _check_positive_int(self.m, "compute.matmul m")
        _check_positive_int(self.n, "compute.matmul n")
        _check_positive_int(self.k, "compute.matmul k")


# ---------------------------------------------------------------------------
# astra.microcode.dma_load — DRAM -> SRAM copy
# ---------------------------------------------------------------------------


@irdl_op_definition
class MicrocodeDmaLoadOp(IRDLOperation):
    """``astra.microcode.dma_load`` — DMA ``dram_ptr`` -> ``sram_ptr``.

    The ``bytes`` property is the byte count to transfer (must be
    ``>= 0``; zero-byte DMAs are emitted by codegen as padding markers
    for memory-alignment guarantees). The source ``dram_ptr`` is an
    untyped ``i64`` host address (DRAM is implicit by name; the
    ``MemorySpace`` enum is reserved for the SRAM-side operands where
    the chip's bank routing matters — spec §9). The destination
    ``sram_ptr`` must carry one of ``<sram0>``, ``<sram1>``, ``<scratch>``
    — DRAM destinations are rejected by the verifier so the
    cycle-accurate simulator's DMA-channel counters stay consistent.
    """

    name = "astra.microcode.dma_load"
    bytes = prop_def(IntAttr)
    dram_ptr = operand_def(IntegerType)
    sram_ptr = result_def(PtrType)

    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = "$dram_ptr attr-dict `:` `(` type($dram_ptr) `)` `->` type($sram_ptr)"

    def __init__(
        self,
        dram_ptr: object,
        sram_ptr_type: PtrType,
        bytes: int | IntAttr,
    ) -> None:
        super().__init__(
            operands=[dram_ptr],
            result_types=[sram_ptr_type],
            properties={"bytes": IntAttr(bytes) if isinstance(bytes, int) else bytes},
        )

    def verify_(self) -> None:
        _check_nonneg_int(self.bytes, "microcode.dma_load bytes")
        _check_dram(self.dram_ptr, "astra.microcode.dma_load source 'dram_ptr'")
        dst = _check_ptr(self.sram_ptr, "astra.microcode.dma_load result 'sram_ptr'")
        if dst.space.data is MemorySpace.DRAM:
            raise VerifyException(
                "astra.microcode.dma_load destination must be an SRAM region "
                "(sram0 | sram1 | scratch), got dram"
            )


# ---------------------------------------------------------------------------
# astra.microcode.dma_store — SRAM -> DRAM copy
# ---------------------------------------------------------------------------


@irdl_op_definition
class MicrocodeDmaStoreOp(IRDLOperation):
    """``astra.microcode.dma_store`` — DMA ``sram_ptr`` -> ``dram_ptr``.

    Mirror of :class:`MicrocodeDmaLoadOp` for the back-direction: the
    source must be an SRAM region and the destination must be DRAM. The
    ``bytes`` property is the transfer count (see :class:`MicrocodeDmaLoadOp`
    for the ``>= 0`` rule).
    """

    name = "astra.microcode.dma_store"
    bytes = prop_def(IntAttr)
    sram_ptr = operand_def(PtrType)
    dram_ptr = result_def(IntegerType)

    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = "$sram_ptr attr-dict `:` `(` type($sram_ptr) `)` `->` type($dram_ptr)"

    def __init__(
        self,
        sram_ptr: object,
        dram_ptr_type: IntegerType,
        bytes: int | IntAttr,
    ) -> None:
        super().__init__(
            operands=[sram_ptr],
            result_types=[dram_ptr_type],
            properties={"bytes": IntAttr(bytes) if isinstance(bytes, int) else bytes},
        )

    def verify_(self) -> None:
        _check_nonneg_int(self.bytes, "microcode.dma_store bytes")
        src = _check_ptr(self.sram_ptr, "astra.microcode.dma_store source 'sram_ptr'")
        _check_dram(self.dram_ptr, "astra.microcode.dma_store result 'dram_ptr'")
        if src.space.data is MemorySpace.DRAM:
            raise VerifyException(
                "astra.microcode.dma_store source must be an SRAM region "
                "(sram0 | sram1 | scratch), got dram"
            )


# ---------------------------------------------------------------------------
# astra.microcode.matmul — systolic matmul on sram tiles
# ---------------------------------------------------------------------------


@irdl_op_definition
class MicrocodeMatmulOp(IRDLOperation):
    """``astra.microcode.matmul`` — systolic matmul on sram tiles.

    Computes ``sram_acc[m, n] += sram_a[m, k] @ sram_b[k, n]`` on the
    16x16 systolic array. ``M``, ``N``, ``K`` are required
    :class:`IntAttr` properties (all ``>= 1``); the optional ``bias``
    operand, when present, is added into ``sram_acc`` BEFORE the matmul
    runs (i.e. the ``accumulate`` semantics match the FusedMLP bias-add
    epilogue in Task 42's lowering).

    All three pointer operands (``sram_a``, ``sram_b``, ``sram_acc``,
    optional ``bias``) must carry an SRAM-region memory space
    (``<sram0>``, ``<sram1>``, or ``<scratch>``) — DRAM pointers are
    rejected by the verifier so the codegen never emits a microcode
    instruction that crosses the DMA boundary.
    """

    name = "astra.microcode.matmul"
    m = prop_def(IntAttr)
    n = prop_def(IntAttr)
    k = prop_def(IntAttr)
    sram_a = operand_def(PtrType)
    sram_b = operand_def(PtrType)
    bias = opt_operand_def(PtrType)
    sram_acc = result_def(PtrType)

    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = (
        "$sram_a `,` $sram_b (`,` $bias^)? attr-dict "
        "`:` `(` type($sram_a) `,` type($sram_b) (`,` type($bias)^)? `)` `->` type($sram_acc)"
    )

    def __init__(
        self,
        sram_a: object,
        sram_b: object,
        sram_acc_type: PtrType,
        m: int | IntAttr,
        n: int | IntAttr,
        k: int | IntAttr,
        bias: object | None = None,
    ) -> None:
        operands = [sram_a, sram_b]
        if bias is not None:
            operands.append(bias)
        super().__init__(
            operands=operands,
            result_types=[sram_acc_type],
            properties={
                "m": IntAttr(m) if isinstance(m, int) else m,
                "n": IntAttr(n) if isinstance(n, int) else n,
                "k": IntAttr(k) if isinstance(k, int) else k,
            },
        )

    def verify_(self) -> None:
        _check_positive_int(self.m, "microcode.matmul m")
        _check_positive_int(self.n, "microcode.matmul n")
        _check_positive_int(self.k, "microcode.matmul k")
        a_t = _check_ptr(self.sram_a, "astra.microcode.matmul operand 'sram_a'")
        b_t = _check_ptr(self.sram_b, "astra.microcode.matmul operand 'sram_b'")
        acc_t = _check_ptr(self.sram_acc, "astra.microcode.matmul result 'sram_acc'")
        for name, t in (("sram_a", a_t), ("sram_b", b_t), ("sram_acc", acc_t)):
            if t.space.data is MemorySpace.DRAM:
                raise VerifyException(
                    f"astra.microcode.matmul {name} must be an SRAM region "
                    f"(sram0 | sram1 | scratch), got dram"
                )
        if self.bias is not None:
            bias_t = _check_ptr(self.bias, "astra.microcode.matmul bias")
            if bias_t.space.data is MemorySpace.DRAM:
                raise VerifyException(
                    "astra.microcode.matmul bias must be an SRAM region "
                    "(sram0 | sram1 | scratch), got dram"
                )


# ---------------------------------------------------------------------------
# astra.microcode.vector_gelu — elementwise GELU on a sram tile
# ---------------------------------------------------------------------------


@irdl_op_definition
class MicrocodeVectorGeluOp(IRDLOperation):
    """``astra.microcode.vector_gelu`` — elementwise GELU on a sram tile.

    Operates in place: ``sram_acc -> sram_acc``. The exact erf form is
    used (PyTorch / ONNX default); the Tanh approximation will land as
    an optional ``approximate`` property in a later batch (it is not
    part of the Wave 9 scope per the Task 40 spec).

    ``sram_acc`` must carry an SRAM memory space (``<sram0>``,
    ``<sram1>``, or ``<scratch>``); DRAM pointers are rejected by the
    verifier.
    """

    name = "astra.microcode.vector_gelu"
    sram_acc = operand_def(PtrType)
    out = result_def(PtrType)

    assembly_format = "$sram_acc attr-dict `:` `(` type($sram_acc) `)` `->` type($out)"

    def __init__(self, sram_acc: object, out_type: PtrType) -> None:
        super().__init__(
            operands=[sram_acc],
            result_types=[out_type],
        )

    def verify_(self) -> None:
        in_t = _check_ptr(self.sram_acc, "astra.microcode.vector_gelu operand 'sram_acc'")
        out_t = _check_ptr(self.out, "astra.microcode.vector_gelu result 'out'")
        if in_t.space.data is MemorySpace.DRAM:
            raise VerifyException(
                "astra.microcode.vector_gelu operand must be an SRAM region "
                "(sram0 | sram1 | scratch), got dram"
            )
        if out_t.space.data is MemorySpace.DRAM:
            raise VerifyException(
                "astra.microcode.vector_gelu result must be an SRAM region "
                "(sram0 | sram1 | scratch), got dram"
            )


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


Astra = Dialect(
    "astra",
    [
        ComputeMatmulOp,
        MicrocodeDmaLoadOp,
        MicrocodeDmaStoreOp,
        MicrocodeMatmulOp,
        MicrocodeVectorGeluOp,
    ],
    [
        MemorySpaceAttr,
        PtrType,
    ],
)
