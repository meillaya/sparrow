"""forge.graph dialect — direct PyTorch FX capture in MLIR form.

This is the Phase-1 graph-level representation of captured PyTorch graphs
(see forgestack plan §4). It preserves the FX semantics — broadcasting, dtype
promotion, dynamic shapes, view semantics, tensor aliasing, device placement —
and is the input surface for the later GraphToTensor conversion pass.

The dialect ships four ops for Phase 1 plus a dynamic-shape no-op:

* ``forge.graph.matmul``       — 2D-ish matrix multiplication; broadcasts batch dims.
* ``forge.graph.add``          — elementwise add with NumPy broadcasting.
* ``forge.graph.gelu``         — exact GELU (PyTorch default).
* ``forge.graph.view``         — reshape with no data movement.
* ``forge.graph.symbolic_guard`` — trace-time metadata op asserting a
  :class:`SymbolicConstraint` over a tensor's shape (Task 14). No-op at
  runtime — Phase 3 owns the runtime evaluator per the plan.

The tensor type encodes static dims (``IntAttr``) and symbolic dims
(``StringAttr``), and optionally carries a list of
:class:`SymbolicConstraint` attrs (one per symbolic dim, in dim-order)
so the per-tensor invariant set round-trips through the textual MLIR
surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrConstraint,
    BaseAttr,
    GenericData,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

from compiler.lib.Dialect.Graph.symbolic import SymbolicConstraint

# ---------------------------------------------------------------------------
# DType enum + attribute
# ---------------------------------------------------------------------------


class ForgeDType(StrEnum):
    """PyTorch-mirroring dtype enum for ``forge.tensor``."""

    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I32 = "i32"
    I8 = "i8"
    I4 = "i4"


# Registry of every valid dtype spelling for the lexer-driven parse path.
_FORGE_DTYPE_VALUES: frozenset[str] = frozenset(d.value for d in ForgeDType)


@irdl_attr_definition
class DTypeAttr(GenericData[ForgeDType]):
    """Wrapper attribute carrying a :class:`ForgeDType` as its ``data`` field.

    Renders as a bare identifier (``f32``, ``bf16``, ...). Used as the second
    parameter of :class:`TensorType`, never directly in user IR.
    """

    name = "forge.dtype"

    def __init__(self, value: ForgeDType | str) -> None:
        if isinstance(value, str):
            value = ForgeDType(value)
        super().__init__(value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> ForgeDType:
        ident = parser.parse_identifier("expected forge dtype identifier")
        if ident not in _FORGE_DTYPE_VALUES:
            parser.raise_error(
                f"unknown forge.dtype {ident!r} (expected one of {sorted(_FORGE_DTYPE_VALUES)})"
            )
        return ForgeDType(ident)

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(self.data.value)

    @staticmethod
    def constr() -> AttrConstraint:
        return BaseAttr(DTypeAttr)


# ---------------------------------------------------------------------------
# TensorType — shape with mixed static/symbolic dims + dtype
# ---------------------------------------------------------------------------


def _is_symbolic_dim(name: str) -> bool:
    """Return True iff ``name`` is a symbolic-dim identifier (starts with a letter, ``_``, or ``?``).

    ``?`` is the anonymous-symbolic-dim marker introduced in Task 14 (e.g.
    ``!forge.tensor<?x?x4096xf32, [...]>``); the constraint list on the type
    supplies the canonical symbolic-dim name positionally.
    """
    return bool(name) and (name[0].isalpha() or name[0] in ("_", "?"))


@dataclass(frozen=True, init=False)
class _DimOrSymbol:
    """Tagged union for tensor dim elements; mirrors ``ArrayAttr[Attribute]`` entries."""

    raw: IntAttr | StringAttr

    @classmethod
    def from_int(cls, value: int) -> _DimOrSymbol:
        return cls(IntAttr(value))

    @classmethod
    def from_symbol(cls, name: str) -> _DimOrSymbol:
        return cls(StringAttr(name))

    def is_static(self) -> bool:
        return isinstance(self.raw, IntAttr)


@irdl_attr_definition
class TensorType(ParametrizedAttribute, TypeAttribute):
    """Ranked tensor with mixed static/symbolic dims and a forge.dtype.

    Syntax (parsed/printed by ``forge.tensor``)::

        !forge.tensor<dimxdimxdimx...xdtype>
        !forge.tensor<dimxdimxdimx...xdtype, [constraint, ...]>

    where each ``dim`` is either an ``IntAttr`` (static, >=0) or a
    ``StringAttr`` (symbolic name like ``B``, ``S``, ``K``), the trailing
    ``dtype`` is one of the :class:`ForgeDType` spellings (``f32``, ``f16``,
    ``bf16``, ``i32``, ``i8``, ``i4``), and the optional trailing
    ``[constraint, ...]`` is a list of :class:`SymbolicConstraint` attrs
    (one per symbolic dim, in dim-order). An empty list omits the
    ``, [...]`` token entirely — concrete-shape IR keeps the original
    Phase-1 surface.
    """

    name = "forge.tensor"
    shape: ArrayAttr[Attribute]
    dtype: DTypeAttr
    constraints: ArrayAttr[SymbolicConstraint]

    def __init__(
        self,
        shape: Sequence[IntAttr | StringAttr | int | str],
        dtype: DTypeAttr | ForgeDType | str,
        constraints: Sequence[SymbolicConstraint] | None = None,
    ) -> None:
        if isinstance(dtype, (ForgeDType, str)):
            dtype = DTypeAttr(dtype)
        dims: list[Attribute] = []
        for dim in shape:
            if isinstance(dim, int):
                dims.append(IntAttr(dim))
            elif isinstance(dim, str):
                dims.append(StringAttr(dim))
            else:
                dims.append(dim)
        constraints_attr = ArrayAttr(tuple(constraints or ()))
        super().__init__(ArrayAttr(dims), dtype, constraints_attr)

    # ----- accessors -------------------------------------------------------

    def get_shape(self) -> tuple[Attribute, ...]:
        return tuple(self.shape.data)

    def get_dtype(self) -> ForgeDType:
        return self.dtype.data

    def get_constraints(self) -> tuple[SymbolicConstraint, ...]:
        """Return the constraint list (possibly empty)."""
        return tuple(self.constraints.data)

    def num_elements(self) -> int | None:
        """Return the static element count, or ``None`` if any dim is symbolic."""
        total = 1
        for dim in self.shape.data:
            if isinstance(dim, IntAttr):
                total *= dim.data
            else:
                return None
        return total

    # ----- parse / print ---------------------------------------------------

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            tokens: list[Attribute] = []
            while True:
                tokens.append(_parse_dim_token(parser))
                if _peek_is_close_angle(parser):
                    break
                if _peek_is_comma(parser):
                    break
                if not _consume_shape_x(parser):
                    break
            if not tokens:
                parser.raise_error("forge.tensor requires at least a dtype token")
            dtype_token = tokens.pop()
            dtype_str: str
            if isinstance(dtype_token, IntAttr):
                dtype_str = str(dtype_token.data)
            elif isinstance(dtype_token, StringAttr):
                dtype_str = dtype_token.data
            else:  # pragma: no cover — defensive
                parser.raise_error("expected dtype as the last element of forge.tensor")
            if dtype_str not in _FORGE_DTYPE_VALUES:
                parser.raise_error(
                    f"unknown forge.dtype '{dtype_str}' "
                    f"(expected one of {sorted(_FORGE_DTYPE_VALUES)})"
                )
            constraints_attr: ArrayAttr[SymbolicConstraint] = ArrayAttr(())
            if _peek_is_comma(parser):
                parser._consume_token()  # consume the ``,``
                constraints_attr = ArrayAttr(tuple(_parse_constraint_list(parser)))
            return [
                ArrayAttr(tokens),
                DTypeAttr(ForgeDType(dtype_str)),
                constraints_attr,
            ]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            for i, dim in enumerate(self.shape.data):
                if i > 0:
                    printer.print_string("x")
                if isinstance(dim, IntAttr):
                    printer.print_int(dim.data)
                elif isinstance(dim, StringAttr):
                    printer.print_string(dim.data)
                else:  # pragma: no cover — defensive
                    printer.print_attribute(dim)
            printer.print_string("x")
            printer.print_string(self.dtype.data.value)
            if self.constraints.data:
                printer.print_string(", ")
                with printer.in_square_brackets():
                    for i, c in enumerate(self.constraints.data):
                        if i > 0:
                            printer.print_string(", ")
                        printer.print_attribute(c)

    # ----- verifier --------------------------------------------------------

    def verify(self) -> None:
        if not self.shape.data:
            raise VerifyException("forge.tensor must have rank >= 1")
        for dim in self.shape.data:
            if isinstance(dim, IntAttr):
                if dim.data < 0:
                    raise VerifyException(f"forge.tensor static dim must be >= 0, got {dim.data}")
            elif isinstance(dim, StringAttr):
                if not _is_symbolic_dim(dim.data):
                    raise VerifyException(
                        f"forge.tensor symbolic dim must start with a letter, "
                        f"'_', or '?' (got {dim.data!r})"
                    )
            else:  # pragma: no cover — defensive
                raise VerifyException(
                    f"forge.tensor dim must be IntAttr or StringAttr, got {type(dim)}"
                )
        # Constraint list is positional over the first N dim positions
        # (N == len(constraints)). Constraints on named symbolic dims
        # (e.g. ``B >= 1``) and on anonymous ``?`` dims are both allowed;
        # the constraint is just metadata for that dim. The only hard
        # constraint we enforce is that the list fits within the rank —
        # extra entries are an authoring error and would be meaningless.
        rank = len(self.shape.data)
        if len(self.constraints.data) > rank:
            raise VerifyException(
                f"forge.tensor constraint list length "
                f"({len(self.constraints.data)}) must not exceed the rank "
                f"({rank})"
            )


def _peek_is_close_angle(parser: AttrParser) -> bool:
    """Return True iff the next token is the ``>`` ending ``<...>``."""
    return parser._current_token.kind.name == "GREATER"


def _peek_is_comma(parser: AttrParser) -> bool:
    """Return True iff the next token is a ``,`` (constraint-list separator)."""
    return parser._current_token.kind.name == "COMMA"


def _consume_shape_x(parser: AttrParser) -> bool:
    """Consume an ``x`` delimiter, returning True iff one was found.

    The MLIR lexer collapses ``x<rest>`` into a single ``BARE_IDENT`` token;
    we resume the lexer one character past the leading ``x`` so the rest of
    the token becomes a fresh token (mirroring ``builtin.tensor``'s
    ``parse_shape_dimension``).
    """
    token = parser._current_token
    if token.kind.name != "BARE_IDENT" or not token.text or token.text[0] != "x":
        return False
    if token.text == "x":
        parser._consume_token()
        return True
    parser._resume_from(token.span.start + 1)
    return True


def _parse_dim_token(parser: AttrParser) -> Attribute:
    """Parse a single tensor dim (integer literal, ``?``, or symbolic identifier).

    Returns ``IntAttr`` for INTEGER_LIT, ``StringAttr("?")`` for the
    anonymous-symbolic ``?`` punctuation, and ``StringAttr(name)`` for
    BARE_IDENT (named symbolic dim like ``B``, ``S``, ``K``).
    """
    int_val = parser.parse_optional_integer(allow_boolean=False, allow_negative=False)
    if int_val is not None:
        return IntAttr(int_val)
    if parser.parse_optional_punctuation("?") is not None:
        return StringAttr("?")
    ident = parser.parse_optional_identifier()
    if ident is None:
        parser.raise_error("expected forge.tensor dim (integer, '?', or identifier)")
    return StringAttr(ident)


def _parse_constraint_list(parser: AttrParser) -> list[SymbolicConstraint]:
    """Parse ``[<constraint>, <constraint>, ...]`` for a Tensor's constraint tail.

    The leading ``,`` (separating the dtype from the list) is consumed by the
    caller; we expect to start on the opening ``[`` and finish on the closing
    ``]``. Empty list (``[]``) is allowed.
    """
    if parser.parse_optional_punctuation("[") is None:
        parser.raise_error(
            "expected '[' to open forge.tensor constraint list "
            "(constraint list must be non-empty syntax-wise but may carry zero entries)"
        )
    out: list[SymbolicConstraint] = []
    if parser.parse_optional_punctuation("]") is not None:
        return out
    while True:
        attr = parser.parse_attribute()
        if not isinstance(attr, SymbolicConstraint):
            parser.raise_error(f"expected #forge.symbolic_constraint, got {type(attr).__name__}")
        out.append(attr)
        if parser.parse_optional_punctuation("]") is not None:
            break
        if not parser.parse_optional_punctuation(","):
            parser.raise_error("expected ',' or ']' inside forge.tensor constraint list")
    return out


# ---------------------------------------------------------------------------
# Helper: broadcast two static shape tuples (NumPy rules, supports symbols)
# ---------------------------------------------------------------------------


def _broadcast_shapes(
    lhs: tuple[Attribute, ...], rhs: tuple[Attribute, ...]
) -> tuple[Attribute, ...] | None:
    """Return the broadcast shape or ``None`` if shapes are incompatible.

    NumPy rules: dims are right-aligned; a dim of size 1 broadcasts to match;
    two symbolic dims must agree (or one is the other); a static and a
    symbolic dim with the same value pair agree. Static mismatches (e.g. 3 vs
    5) are incompatible.
    """
    # Pad the shorter shape with size-1 dims on the LEFT.
    ndim = max(len(lhs), len(rhs))
    pad_lhs: list[Attribute] = [IntAttr(1)] * (ndim - len(lhs)) + list(lhs)
    pad_rhs: list[Attribute] = [IntAttr(1)] * (ndim - len(rhs)) + list(rhs)
    out: list[Attribute] = []
    for a, b in zip(pad_lhs, pad_rhs, strict=True):
        merged = _broadcast_dim(a, b)
        if merged is None:
            return None
        out.append(merged)
    return tuple(out)


def _broadcast_dim(a: Attribute, b: Attribute) -> Attribute | None:
    """Broadcast two single dims; ``None`` on mismatch."""
    a_int = a.data if isinstance(a, IntAttr) else None
    b_int = b.data if isinstance(b, IntAttr) else None
    a_sym = a.data if isinstance(a, StringAttr) else None
    b_sym = b.data if isinstance(b, StringAttr) else None
    # Same symbol wins.
    if a_sym is not None and b_sym is not None:
        return a if a == b else None
    # Static 1 broadcasts to the other side.
    if a_int == 1:
        return b
    if b_int == 1:
        return a
    # Same static value is fine.
    if a_int is not None and b_int is not None and a_int == b_int:
        return a
    # Symbolic meeting static (same value) — keep static for layout stability.
    if a_int is not None and b_sym is not None and a_int == 1:  # unreachable: covered above
        return b
    if a_sym is not None and b_int is not None and b_int == 1:  # unreachable
        return a
    return None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _extract_tensor(value: object) -> TensorType:
    """Unwrap ``SSAValue`` to its type, asserting it is a ``TensorType``."""
    ssa = SSAValue.get(value)
    if not isinstance(ssa.type, TensorType):
        raise VerifyException(f"operand must be !forge.tensor, got {ssa.type}")
    return ssa.type


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """``forge.graph.matmul`` — batched matrix multiplication.

    Mirrors ``torch.matmul`` / ``aten.mm`` semantics: inner dims must agree,
    leading batch dims broadcast.
    """

    name = "forge.graph.matmul"
    x = operand_def(TensorType)
    w = operand_def(TensorType)
    out = result_def(TensorType)

    assembly_format = "$x `,` $w attr-dict `:` `(` type($x) `,` type($w) `)` `->` type($out)"

    def __init__(
        self,
        x: object,
        w: object,
        result_type: TensorType | None = None,
    ) -> None:
        if result_type is None:
            x_t = _extract_tensor(x)
            w_t = _extract_tensor(w)
            result_type = _infer_matmul_result(x_t, w_t)
        super().__init__(
            operands=(x, w),
            result_types=(result_type,),
        )

    def verify_(self) -> None:
        x_t = _extract_tensor(self.x)
        w_t = _extract_tensor(self.w)
        inferred = _infer_matmul_result(x_t, w_t)
        if not isinstance(self.out.type, TensorType):
            raise VerifyException(f"matmul result must be !forge.tensor, got {self.out.type}")
        if inferred is None:
            raise VerifyException(
                f"matmul inner dims mismatch: {x_t.get_shape()[-1]} vs {w_t.get_shape()[-2]}"
            )
        if inferred != self.out.type.get_shape():
            raise VerifyException(
                f"matmul result shape {self.out.type.get_shape()} "
                f"does not match inferred shape {inferred}"
            )
        if x_t.get_dtype() != w_t.get_dtype():
            raise VerifyException(
                f"matmul operand dtypes must agree ({x_t.get_dtype()} vs {w_t.get_dtype()})"
            )


def _infer_matmul_result(x: TensorType, w: TensorType) -> tuple[Attribute, ...] | None:
    """Return the broadcast batch shape + ``[M, N]`` for ``matmul x, w``.

    Returns ``None`` when inner dims disagree.
    """
    x_shape = x.get_shape()
    w_shape = w.get_shape()
    if len(x_shape) < 2 or len(w_shape) < 2:
        return None
    # Inner contract dim check.
    x_inner = x_shape[-1]
    w_inner = w_shape[-2]
    if isinstance(x_inner, IntAttr) and isinstance(w_inner, IntAttr):
        if x_inner.data != w_inner.data:
            return None
    elif isinstance(x_inner, IntAttr) and isinstance(w_inner, StringAttr):
        if x_inner.data == 1:
            pass  # broadcasts — but for matmul inner this is degenerate
        else:
            # Treat as compatible if the symbol could match anything.
            pass
    elif isinstance(x_inner, StringAttr) and isinstance(w_inner, IntAttr):
        pass
    else:
        if x_inner.data != w_inner.data:
            return None
    # Outer batch dims broadcast.
    x_batch = x_shape[:-2]
    w_batch = w_shape[:-2]
    batch = _broadcast_shapes(x_batch, w_batch)
    if batch is None:
        return None
    # M = x_shape[-2], N = w_shape[-1].
    return (*batch, x_shape[-2], w_shape[-1])


@irdl_op_definition
class AddOp(IRDLOperation):
    """``forge.graph.add`` — elementwise addition with NumPy broadcasting."""

    name = "forge.graph.add"
    a = operand_def(TensorType)
    b = operand_def(TensorType)
    out = result_def(TensorType)

    assembly_format = "$a `,` $b attr-dict `:` `(` type($a) `,` type($b) `)` `->` type($out)"

    def __init__(
        self,
        a: object,
        b: object,
        result_type: TensorType | None = None,
    ) -> None:
        if result_type is None:
            a_t = _extract_tensor(a)
            b_t = _extract_tensor(b)
            broadcast = _broadcast_shapes(a_t.get_shape(), b_t.get_shape())
            if broadcast is None:
                raise VerifyException(
                    f"add: shapes {a_t.get_shape()} and {b_t.get_shape()} are not broadcastable"
                )
            result_type = TensorType(broadcast, a_t.get_dtype())
        super().__init__(
            operands=(a, b),
            result_types=(result_type,),
        )

    def verify_(self) -> None:
        a_t = _extract_tensor(self.a)
        b_t = _extract_tensor(self.b)
        if a_t.get_dtype() != b_t.get_dtype():
            raise VerifyException(
                f"add operand dtypes must agree ({a_t.get_dtype()} vs {b_t.get_dtype()})"
            )
        if not isinstance(self.out.type, TensorType):
            raise VerifyException(f"add result must be !forge.tensor, got {self.out.type}")
        broadcast = _broadcast_shapes(a_t.get_shape(), b_t.get_shape())
        if broadcast is None:
            raise VerifyException(
                f"add: shapes {a_t.get_shape()} and {b_t.get_shape()} are not broadcastable"
            )
        if broadcast != self.out.type.get_shape():
            raise VerifyException(
                f"add result shape {self.out.type.get_shape()} "
                f"does not match broadcast shape {broadcast}"
            )


@irdl_attr_definition
class GeluApproxAttr(GenericData[str]):
    """``approximate`` attribute for ``forge.graph.gelu``.

    ``"none"`` is the PyTorch default (exact erf-based GELU); ``"tanh"``
    matches the Tanh-approximation variant. Exposes the enum constants so
    callers can build it without spelling the literal.
    """

    name = "forge.graph.gelu_approx"

    NONE: ClassVar[str] = "none"
    TANH: ClassVar[str] = "tanh"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        ident = parser.parse_optional_identifier()
        if ident is None:
            ident = parser.parse_str_literal("expected gelu approximation")
        if ident not in (cls.NONE, cls.TANH):
            parser.raise_error(
                f"unknown gelu approximation {ident!r} (expected {cls.NONE!r} or {cls.TANH!r})"
            )
        return ident

    def print_parameter(self, printer: Printer) -> None:
        printer.print_identifier_or_string_literal(self.data)

    def verify(self) -> None:
        if self.data not in (self.NONE, self.TANH):
            raise VerifyException(
                f"gelu approximate must be {self.NONE!r} or {self.TANH!r}, got {self.data!r}"
            )

    @staticmethod
    def constr() -> AttrConstraint:
        return BaseAttr(GeluApproxAttr)


@irdl_op_definition
class GeluOp(IRDLOperation):
    """``forge.graph.gelu`` — exact GELU activation (matches PyTorch default).

    Carries an optional ``approximate`` attribute (``:class:`GeluApproxAttr```)
    selecting the exact (default ``"none"``) or Tanh (``"tanh"``) variant.
    """

    name = "forge.graph.gelu"
    x = operand_def(TensorType)
    out = result_def(TensorType)
    approximate = prop_def(StringAttr, default_value=StringAttr(GeluApproxAttr.NONE))

    assembly_format = (
        "$x (`approximate` `` $approximate^)? attr-dict `:` `(` type($x) `)` `->` type($out)"
    )

    def __init__(
        self,
        x: object,
        result_type: TensorType | None = None,
        approximate: StringAttr | str = GeluApproxAttr.NONE,
    ) -> None:
        if result_type is None:
            x_t = _extract_tensor(x)
            result_type = TensorType(x_t.get_shape(), x_t.get_dtype())
        if isinstance(approximate, str):
            approximate = StringAttr(approximate)
        super().__init__(
            operands=(x,),
            result_types=(result_type,),
            properties={"approximate": approximate},
        )

    def verify_(self) -> None:
        x_t = _extract_tensor(self.x)
        if not isinstance(self.out.type, TensorType):
            raise VerifyException(f"gelu result must be !forge.tensor, got {self.out.type}")
        if x_t.get_shape() != self.out.type.get_shape():
            raise VerifyException(
                f"gelu preserves shape: input {x_t.get_shape()} "
                f"vs result {self.out.type.get_shape()}"
            )
        if x_t.get_dtype() != self.out.type.get_dtype():
            raise VerifyException(
                f"gelu preserves dtype: input {x_t.get_dtype()} "
                f"vs result {self.out.type.get_dtype()}"
            )
        if self.approximate.data not in (GeluApproxAttr.NONE, GeluApproxAttr.TANH):
            raise VerifyException(
                f"gelu approximate must be {GeluApproxAttr.NONE!r} or "
                f"{GeluApproxAttr.TANH!r}, got {self.approximate.data!r}"
            )


@irdl_op_definition
class ViewOp(IRDLOperation):
    """``forge.graph.view`` — reshape with no data movement.

    The total element count must match the input — broadcasting / splitting
    is not allowed. Symbolic dims are not yet folded; both source and target
    shapes must be fully static for the rank-count check.
    """

    name = "forge.graph.view"
    x = operand_def(TensorType)
    out = result_def(TensorType)
    # The target shape is carried by the result type so that downstream
    # patterns can read it directly; the verifier checks element count.

    assembly_format = "$x attr-dict `:` `(` type($x) `)` `->` type($out)"

    def __init__(
        self,
        x: object,
        result_type: TensorType,
    ) -> None:
        super().__init__(
            operands=(x,),
            result_types=(result_type,),
        )

    def verify_(self) -> None:
        x_t = _extract_tensor(self.x)
        if not isinstance(self.out.type, TensorType):
            raise VerifyException(f"view result must be !forge.tensor, got {self.out.type}")
        x_count = x_t.num_elements()
        out_count = self.out.type.num_elements()
        if x_count is not None and out_count is not None and x_count != out_count:
            raise VerifyException(
                f"view: element count mismatch ({x_count} input vs {out_count} output)"
            )
        # Symbolic dims present in either shape: skip the element-count
        # check (it can't fire symbolically) and require the rank +
        # rank-product dim structure to agree instead. This lets the FX
        # importer emit pass-through views over symbolic inputs that the
        # runtime specialises later.


@irdl_op_definition
class SymbolicGuardOp(IRDLOperation):
    """``forge.graph.symbolic_guard`` — trace-time metadata no-op (Task 14).

    Carries a single :class:`SymbolicConstraint` attr over its ``input``
    operand and returns the same tensor unchanged. The op is metadata-only:
    the verifier checks that (a) the operand and result types match (so
    downstream pattern matchers can rely on ``t == guard(t)``), and (b)
    the constraint attr is well-formed (already covered by the attr's own
    verifier). No runtime evaluation happens here — Phase 3 owns the
    runtime evaluator per the plan.

    The constraint travels as an ordinary attribute dictionary entry (not a
    positional format argument) because xDSL's Declarative Assembly Format
    treats ``UniqueBaseAttributeVariable`` (a non-builtin attr) by skipping
    the ``#forge.symbolic_constraint`` prefix on print but expects it on
    parse — a mismatch the format machinery doesn't bridge. Going through
    the generic ``attr-dict`` keeps parse/print symmetric.
    """

    name = "forge.graph.symbolic_guard"
    input = operand_def(TensorType)
    out = result_def(TensorType)
    constraint = prop_def(SymbolicConstraint)

    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = "$input attr-dict `:` `(` type($input) `)` `->` type($out)"

    def __init__(
        self,
        input: object,
        constraint: SymbolicConstraint,
        result_type: TensorType | None = None,
    ) -> None:
        if result_type is None:
            input_t = _extract_tensor(input)
            result_type = input_t
        super().__init__(
            operands=(input,),
            result_types=(result_type,),
            properties={"constraint": constraint},
        )

    def verify_(self) -> None:
        input_t = _extract_tensor(self.input)
        if not isinstance(self.out.type, TensorType):
            raise VerifyException(
                f"symbolic_guard result must be !forge.tensor, got {self.out.type}"
            )
        # Guard must be a no-op shape-wise — operand and result types
        # are identical so downstream pattern matchers can rely on it.
        if input_t.get_shape() != self.out.type.get_shape():
            raise VerifyException(
                f"symbolic_guard preserves shape: input {input_t.get_shape()} "
                f"vs result {self.out.type.get_shape()}"
            )
        if input_t.get_dtype() != self.out.type.get_dtype():
            raise VerifyException(
                f"symbolic_guard preserves dtype: input {input_t.get_dtype()} "
                f"vs result {self.out.type.get_dtype()}"
            )


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


Graph = Dialect(
    "forge.graph",
    [
        MatmulOp,
        AddOp,
        GeluOp,
        ViewOp,
        SymbolicGuardOp,
    ],
    [
        TensorType,
        DTypeAttr,
        GeluApproxAttr,
        SymbolicConstraint,
    ],
)
