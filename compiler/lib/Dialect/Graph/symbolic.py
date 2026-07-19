"""forge.graph symbolic-shape constraint attribute (Task 14).

Spec §3 calls out the runtime invariants we generate guards for::

    B >= 1
    S >= 1
    K == 4096
    N == 11008
    input.stride[-1] == 1

We model those as a single :class:`SymbolicConstraint` ParametrizedAttribute
with a ``(name, kind, value)`` triple, so every constraint fits in one MLIR
attribute::

    #forge.symbolic_constraint<"B", greater_equal, 1>
    #forge.symbolic_constraint<"S", greater_equal, 1>
    #forge.symbolic_constraint<"K", equal, 4096>

The :class:`TensorType` (defined in the parent :mod:`compiler.lib.Dialect.Graph`
module) optionally carries an :class:`ArrayAttr` of these constraints in
dim-order — one entry per symbolic dim. The :class:`SymbolicGuardOp` op uses a
single constraint attr (the constraint it asserts at trace-time). Both are
metadata-only at this stage; the runtime evaluator lands in Phase 3 per the
plan.
"""

from __future__ import annotations

from xdsl.dialects.builtin import IntAttr, IntegerAttr, StringAttr
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class SymbolicConstraintKind(StrEnum):
    """Constraint kinds supported by :class:`SymbolicConstraint`.

    Each kind pairs with an :class:`IntAttr` value:

    * ``greater_equal`` — ``dim >= value``  (e.g. ``B >= 1``).
    * ``equal``         — ``dim == value``  (e.g. ``K == 4096``).
    * ``divisible``     — ``dim % value == 0`` (e.g. ``S % 16 == 0``).
    * ``less_equal``    — ``dim <= value``  (e.g. ``N <= 11008``).
    * ``range``         — ``dim == value`` (placeholder until dynamic
      upper-bound expressions land; identical semantic to ``equal`` for now
      but is rendered distinctly so the lowering pipeline can pattern-match
      on it later).
    """

    GREATER_EQUAL = "greater_equal"
    EQUAL = "equal"
    DIVISIBLE = "divisible"
    LESS_EQUAL = "less_equal"
    RANGE = "range"


@irdl_attr_definition
class SymbolicConstraintKindAttr(
    EnumAttribute[SymbolicConstraintKind], SpacedOpaqueSyntaxAttribute
):
    """``#forge.symbolic_constraint_kind<kind>`` — enum wrapper for the kind.

    Renders as ``#forge.symbolic_constraint_kind<greater_equal>`` etc. Lives
    as its own attribute so :class:`SymbolicConstraint` can carry it as a
    multi-param attribute (xDSL ``EnumAttribute`` is a single-param ``Data``).
    """

    name = "forge.symbolic_constraint_kind"


@irdl_attr_definition
class SymbolicConstraint(ParametrizedAttribute):
    """``#forge.symbolic_constraint<name, kind, value>`` — symbolic-dim invariant.

    Renders as ``#forge.symbolic_constraint<"B", greater_equal, 1>`` and is
    both the per-tensor entry in :class:`TensorType`'s constraint list and
    the :attr:`constraint` attr of :class:`SymbolicGuardOp`. The ``name``
    slot is the symbolic-dim identifier (``B``, ``S``, ``K``) — it lets
    the textual MLIR display ``[B >= 1, S >= 1]`` directly.
    """

    name = "forge.symbolic_constraint"
    name_field: StringAttr
    kind: SymbolicConstraintKindAttr
    value: IntAttr

    def __init__(
        self,
        name: str | StringAttr,
        kind: SymbolicConstraintKind | str | SymbolicConstraintKindAttr,
        value: int | IntAttr | IntegerAttr,
    ) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        if isinstance(kind, (SymbolicConstraintKind, str)):
            kind = SymbolicConstraintKindAttr(
                SymbolicConstraintKind(kind) if isinstance(kind, str) else kind
            )
        if isinstance(value, IntegerAttr):
            value = IntAttr(int(value.value.data))
        elif isinstance(value, int):
            value = IntAttr(value)
        super().__init__(name, kind, value)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        # The caller (`_parse_extended_type_or_attribute_body`) leaves us
        # with the opening `<` still in the token stream; consume it.
        with parser.in_angle_brackets():
            name_token = parser.parse_attribute()
            if not isinstance(name_token, StringAttr):
                parser.raise_error(
                    f"symbolic_constraint name must be a string literal, "
                    f"got {type(name_token).__name__}"
                )
                raise  # pragma: no cover
            if not parser.parse_optional_punctuation(","):
                parser.raise_error(
                    "expected ',' after symbolic constraint name, then kind, value"
                )
            kind_ident = parser.parse_identifier(
                "expected symbolic constraint kind identifier"
            )
            try:
                kind_attr_obj = SymbolicConstraintKindAttr(
                    SymbolicConstraintKind(kind_ident)
                )
            except ValueError:
                parser.raise_error(
                    f"unknown symbolic constraint kind {kind_ident!r} "
                    f"(expected one of "
                    f"{[k.value for k in SymbolicConstraintKind]})"
                )
                raise  # pragma: no cover
            if not parser.parse_optional_punctuation(","):
                parser.raise_error(
                    "expected ',' after symbolic constraint kind, then an integer value"
                )
            value_token = parser.parse_attribute()
            int_value: int
            if isinstance(value_token, IntAttr):
                int_value = value_token.data
            elif isinstance(value_token, IntegerAttr):
                int_value = int(value_token.value.data)
            else:
                parser.raise_error(
                    f"symbolic_constraint value must be an integer literal, "
                    f"got {type(value_token).__name__}"
                )
                raise  # pragma: no cover
            if int_value < 0:
                parser.raise_error(
                    f"symbolic_constraint value must be >= 0, got {int_value}"
                )
            value_attr: Attribute = IntAttr(int_value)
        return [name_token, kind_attr_obj, value_attr]

    def print_parameters(self, printer: Printer) -> None:
        # Mirror `parse_parameters`: we own the `< ... >` brackets because
        # the default `print_paramattr_parameters` would emit
        # `<<...>, <...>, <...>>` (double-bracketed) for our three-param shape.
        with printer.in_angle_brackets():
            printer.print_string(f'"{self.name_field.data}"')
            printer.print_string(", ")
            printer.print_string(self.kind.data.value)
            printer.print_string(", ")
            printer.print_int(self.value.data)

    def verify(self) -> None:
        if not isinstance(self.name_field, StringAttr):
            raise VerifyException(
                f"symbolic_constraint name must be a StringAttr, got "
                f"{type(self.name_field).__name__}"
            )
        if not _is_symbolic_dim(self.name_field.data):
            raise VerifyException(
                f"symbolic_constraint name must start with a letter or '_', "
                f"got {self.name_field.data!r}"
            )
        if not isinstance(self.value, IntAttr):
            raise VerifyException(
                f"symbolic_constraint value must be an IntAttr, got "
                f"{type(self.value).__name__}"
            )
        if self.value.data < 0:
            raise VerifyException(
                f"symbolic_constraint value must be >= 0, got {self.value.data}"
            )


def _is_symbolic_dim(name: str) -> bool:
    """Return True iff ``name`` is a valid symbolic-dim identifier."""
    return bool(name) and (name[0].isalpha() or name[0] == "_")