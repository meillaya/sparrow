# compiler/lib/Dialect/Schedule/__init__.py
#
# `forge.schedule` xDSL dialect — Wave 0 / Task 2.
#
# This is the metadata-carrier dialect that Phase 4 (Astra tiling) and Phase 7
# (heterogeneous partitioning) will read from. It does **NOT** lower
# anything yet: the only op (``forge.schedule.tiled_gemm``) is a STUB that
# carries the three pieces of scheduling metadata over the IR boundary so
# downstream phases can pattern-match on it without re-deriving shape / loop
# / pipeline knobs from external annotations.
#
# Attributes:
#   * ``#forge.tile_shape<MxNxK>``        — three positive ints (M, N, K)
#   * ``#forge.loop_order<NAME>``         — one of mnk, mkn, nmk, nkm, kmn, knm
#   * ``#forge.pipeline_stages<N>``       — software-pipeline depth, N >= 1
#
# Operations:
#   * ``forge.schedule.tiled_gemm`` — STUB: metadata-only matmul with tile /
#     loop / stages. Operand/result shapes mirror ``forge.tensor.gemm`` but
#     the inner shape math is **not** lowered — that belongs to Phase 4.

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import IntAttr
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

from compiler.lib.Dialect.Graph import TensorType

# ---------------------------------------------------------------------------
# LoopOrder enum + attribute
# ---------------------------------------------------------------------------


class LoopOrder(StrEnum):
    """Loop-nesting order for the tiled GEMM (Phase 4 Astra target).

    The mnemonic is a permutation of ``{M, N, K}`` where the first letter
    is the outermost loop and the last is the innermost contiguous
    dimension. Six permutations are recognised; users are expected to
    pick the one that best matches the target microarchitecture (e.g.
    ``kmn`` on a row-major A / col-major B matmul).
    """

    MNK = "mnk"
    MKN = "mkn"
    NMK = "nmk"
    NKM = "nkm"
    KMN = "kmn"
    KNM = "knm"


@irdl_attr_definition
class LoopOrderAttr(EnumAttribute[LoopOrder]):
    """``#forge.loop_order<NAME>`` — wraps a :class:`LoopOrder` in ``<...>``.

    Mirrors :class:`compiler.lib.Dialect.Tensor.ActivationAttr` — the
    base ``EnumAttribute`` does not print angle brackets, so we override
    ``print_parameter`` / ``parse_parameter`` to wrap the enum literal.
    """

    name = "forge.loop_order"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_identifier_or_string_literal(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> LoopOrder:
        with parser.in_angle_brackets():
            ident = parser.parse_identifier()
            try:
                return LoopOrder(ident)
            except ValueError as exc:
                valid = ", ".join(repr(m.value) for m in LoopOrder)
                parser.raise_error(f"unknown loop_order {ident!r}; expected one of: {valid}")
                # raise_error raises; the line below is unreachable but keeps
                # the type checker happy.
                raise VerifyException(  # pragma: no cover
                    f"unknown loop_order {ident!r}"
                ) from exc


# ---------------------------------------------------------------------------
# Helpers shared between tile_shape / pipeline_stages parsing
# ---------------------------------------------------------------------------


def _parse_pos_int(parser: AttrParser, what: str) -> int:
    """Parse a non-negative integer (no booleans, no negatives).

    Raises a parse error if the next token is not an integer literal.
    The verifier downstream (``verify``) is responsible for rejecting 0
    — this helper only ensures syntactic shape ``<...>``.
    """
    val = parser.parse_optional_integer(allow_boolean=False, allow_negative=False)
    if val is None:
        parser.raise_error(f"expected non-negative integer for {what}")
    return val  # type: ignore[return-value]  (parser.raise_error never returns)


def _consume_split_x(parser: AttrParser, context_msg: str) -> None:
    """Consume the ``x`` separator between dim tokens.

    The MLIR lexer sometimes glues ``x`` to the following digits into a
    single ``BARE_IDENT`` token (e.g. ``<64x128x32>`` lexes as
    ``64`` ``x128x32`` ``>``); we mirror
    ``compiler.lib.Dialect.Graph._consume_shape_x`` and rewind the lexer
    by one character in that case so the next digit becomes its own
    ``INTEGER_LIT``.
    """
    token = parser._current_token
    if token.kind.name == "BARE_IDENT" and token.text and token.text[0] == "x":
        if len(token.text) == 1:
            parser._consume_token()
            return
        parser._resume_from(token.span.start + 1)
        return
    parser.raise_error(f"expected 'x' separator in {context_msg}")


# ---------------------------------------------------------------------------
# TileShape — three positive ints
# ---------------------------------------------------------------------------


@irdl_attr_definition
class TileShape(ParametrizedAttribute):
    """``#forge.tile_shape<MxNxK>`` — the 3D GEMM tile shape, all dims >= 1.

    ``M`` is the row tile height, ``N`` the column tile width, and ``K``
    the reduction tile depth. All three must be strictly positive — the
    verifier rejects 0 / negative tile dims up-front so downstream
    tile-loop code generation (Phase 4) can assume ``tile.dim >= 1``.
    """

    name = "forge.tile_shape"
    m: IntAttr
    n: IntAttr
    k: IntAttr

    def __init__(
        self,
        m: int | IntAttr,
        n: int | IntAttr,
        k: int | IntAttr,
    ) -> None:
        super().__init__(
            IntAttr(m) if isinstance(m, int) else m,
            IntAttr(n) if isinstance(n, int) else n,
            IntAttr(k) if isinstance(k, int) else k,
        )

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            m = IntAttr(_parse_pos_int(parser, "tile_shape.m"))
            _consume_split_x(parser, "tile_shape<MxNxK>")
            n = IntAttr(_parse_pos_int(parser, "tile_shape.n"))
            _consume_split_x(parser, "tile_shape<MxNxK>")
            k = IntAttr(_parse_pos_int(parser, "tile_shape.k"))
            return (m, n, k)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.m.data)
            printer.print_string("x")
            printer.print_int(self.n.data)
            printer.print_string("x")
            printer.print_int(self.k.data)

    def verify(self) -> None:
        for name, attr in (("m", self.m), ("n", self.n), ("k", self.k)):
            if attr.data < 1:
                raise VerifyException(f"tile dimensions must be positive ({name}={attr.data})")


# ---------------------------------------------------------------------------
# PipelineStages — one positive int
# ---------------------------------------------------------------------------


@irdl_attr_definition
class PipelineStages(ParametrizedAttribute):
    """``#forge.pipeline_stages<N>`` — software-pipeline depth, ``N >= 1``.

    ``N == 1`` means no software pipelining (single-stage). Larger
    values insert ``prologue / steady-state / epilogue`` phases; the
    exact cost-model lives in Phase 7, this attribute only carries
    the user-facing knob.
    """

    name = "forge.pipeline_stages"
    n: IntAttr

    def __init__(self, n: int | IntAttr) -> None:
        super().__init__(IntAttr(n) if isinstance(n, int) else n)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            n = IntAttr(_parse_pos_int(parser, "pipeline_stages.n"))
            return (n,)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.n.data)

    def verify(self) -> None:
        if self.n.data < 1:
            raise VerifyException(f"pipeline stages must be at least 1, got {self.n.data}")


# ---------------------------------------------------------------------------
# tiled_gemm operation (STUB — no lowering until Phase 4)
# ---------------------------------------------------------------------------


def _check_tensor(value: SSAValue, what: str) -> TensorType:
    """Assert that an SSA value is a ``forge.tensor`` and return its type."""
    if not isinstance(value.type, TensorType):
        raise VerifyException(f"{what} must be !forge.tensor, got {value.type}")
    return value.type


def _static_dim(attr: Attribute) -> int | None:
    """Return the static value of a tensor dim, or ``None`` if symbolic."""
    return attr.data if isinstance(attr, IntAttr) else None


@irdl_op_definition
class TiledGemmOp(IRDLOperation):
    """``forge.schedule.tiled_gemm`` — STUB.

    Carries tile / loop / pipeline-stages metadata for a matmul that
    Phase 4 (Astra tiling) and Phase 7 (partitioning) will consume. No
    lowering is implemented yet: this op is a **structural marker** so
    subsequent passes can pattern-match on its metadata without
    re-deriving them from external annotations.

    Operand/result semantics mirror :class:`forge.tensor.gemm` shape
    inference (``[..., M, K]`` op ``[..., K, N]`` -> ``[..., M, N]``) but
    the verifier here intentionally only rejects type mismatches and
    obvious static-inner-K mismatches — the lowering passes own the
    detailed shape rules.
    """

    name = "forge.schedule.tiled_gemm"
    tile = prop_def(TileShape)
    loop = opt_prop_def(
        LoopOrderAttr,
        default_value=LoopOrderAttr(LoopOrder.KMN),
    )
    stages = opt_prop_def(
        PipelineStages,
        default_value=PipelineStages(3),
    )
    a = operand_def(TensorType)
    b = operand_def(TensorType)
    out = result_def(TensorType)

    # `tile` is required; `loop` / `stages` are optional with defaults.
    # Route all three through attr-dict so the assembly format stays simple.
    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = "$a `,` $b attr-dict `:` `(` type($a) `,` type($b) `)` `->` type($out)"

    def __init__(
        self,
        a: object,
        b: object,
        tile: TileShape,
        loop: LoopOrderAttr | LoopOrder | str = LoopOrder.KMN,
        stages: PipelineStages | int = 3,
        result_type: TensorType | None = None,
    ) -> None:
        if isinstance(loop, (LoopOrder, str)):
            loop = LoopOrderAttr(LoopOrder(loop) if isinstance(loop, str) else loop)
        if isinstance(stages, int):
            stages = PipelineStages(stages)
        if result_type is None:
            a_t = _check_tensor(SSAValue.get(a), "forge.schedule.tiled_gemm operand 'a'")
            b_t = _check_tensor(SSAValue.get(b), "forge.schedule.tiled_gemm operand 'b'")
            a_shape = a_t.get_shape()
            b_shape = b_t.get_shape()
            if len(a_shape) < 2 or len(b_shape) < 2:
                raise VerifyException(
                    "forge.schedule.tiled_gemm requires operands with rank >= 2, "
                    f"got a.shape={a_shape}, b.shape={b_shape}"
                )
            # Standard matmul result: batch(a) + (M from a, N from b).
            result_type = TensorType((*a_shape[:-1], b_shape[-1]), a_t.get_dtype())
        super().__init__(
            operands=[a, b],
            result_types=[result_type],
            properties={"tile": tile, "loop": loop, "stages": stages},
        )

    def verify_(self) -> None:
        a_t = _check_tensor(self.a, "forge.schedule.tiled_gemm operand 'a'")
        b_t = _check_tensor(self.b, "forge.schedule.tiled_gemm operand 'b'")
        if not isinstance(self.out.type, TensorType):
            raise VerifyException(
                f"forge.schedule.tiled_gemm result must be !forge.tensor, got {self.out.type}"
            )
        if a_t.get_dtype() != b_t.get_dtype():
            raise VerifyException(
                "forge.schedule.tiled_gemm operand dtypes must match: "
                f"{a_t.get_dtype()} vs {b_t.get_dtype()}"
            )
        # Best-effort inner-K check: only when both dims are static ints.
        a_shape = a_t.get_shape()
        b_shape = b_t.get_shape()
        if len(a_shape) >= 2 and len(b_shape) >= 2:
            a_k = _static_dim(a_shape[-1])
            b_k = _static_dim(b_shape[-2])
            if a_k is not None and b_k is not None and a_k != b_k:
                raise VerifyException(
                    "forge.schedule.tiled_gemm inner K mismatch: "
                    f"a.shape[-1]={a_k} vs b.shape[-2]={b_k}"
                )


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


Schedule = Dialect(
    "forge.schedule",
    [
        TiledGemmOp,
    ],
    [
        TileShape,
        LoopOrderAttr,
        PipelineStages,
    ],
)
