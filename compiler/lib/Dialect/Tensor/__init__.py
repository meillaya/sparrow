# compiler/lib/Dialect/Tensor/__init__.py
#
# `forge.tensor` xDSL dialect — compiler-friendly tensor ops.
#
# Spec §4: the `forge.tensor` dialect is the "lower" level below `forge.graph`.
# It carries explicit shapes, element types, layouts, memory spaces, and target
# metadata; PyTorch-specific autograd / dispatch semantics do NOT live here
# (those live in `forge.graph` and are lowered away by the GraphToTensor pass).
#
# Known debt (tracked separately):
#   * This dialect re-uses `compiler.lib.Dialect.Graph.TensorType` for its
#     operand/result types — `forge.tensor` is defined in the Graph dialect
#     because Task 5 owned that dialect file first. Task 8 will move the
#     type into a shared `compiler/lib/Type/` package and both dialects
#     will reference it. Until then this module imports from `Dialect.Graph`
#     so the round-trip test exercises the canonical `forge.tensor` syntax.

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import BoolAttr
from xdsl.ir import Dialect, EnumAttribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import NoMemoryEffect, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

from compiler.lib.Dialect.Graph import TensorType

# ---------------------------------------------------------------------------
# Activation enum + attribute
# ---------------------------------------------------------------------------


class Activation(StrEnum):
    """PyTorch activation set used by fused epilogues (spec §3)."""

    NONE = "none"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


@irdl_attr_definition
class ActivationAttr(EnumAttribute[Activation]):
    """
    `#forge.activation<VALUE>` where VALUE is one of ``none``, ``relu``,
    ``gelu``, ``silu``. We override ``print_parameter`` / ``parse_parameter``
    to wrap the enum in angle brackets — without that override the default
    EnumAttribute would print `#forge.activationgelu` (no brackets) since
    EnumAttribute does not inherit from OpaqueSyntaxAttribute.
    """

    name = "forge.activation"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_identifier_or_string_literal(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> Activation:
        with parser.in_angle_brackets():
            ident = parser.parse_identifier()
            try:
                return Activation(ident)
            except ValueError as exc:
                valid = ", ".join(repr(m.value) for m in Activation)
                parser.raise_error(f"unknown activation {ident!r}; expected one of: {valid}")
                # raise_error raises; the line below is unreachable but keeps
                # the type checker happy.
                raise VerifyException(  # pragma: no cover
                    f"unknown activation {ident!r}"
                ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_tensor(value: SSAValue, what: str) -> TensorType:
    """Assert that an SSA value is a `forge.tensor` and return its type."""
    if not isinstance(value.type, TensorType):
        raise VerifyException(f"{what} must be !forge.tensor, got {value.type}")
    return value.type


def _broadcast_shape(a_shape: Sequence[int], b_shape: Sequence[int]) -> tuple[int, ...] | None:
    """
    Compute the broadcast of two static shapes, NumPy-style, or return None
    if the shapes cannot be broadcast. Empty shape ``()`` broadcasts to any
    shape; mismatched non-1 dims are an error.
    """
    rev_a = list(reversed(a_shape))
    rev_b = list(reversed(b_shape))
    out: list[int] = []
    for i in range(max(len(rev_a), len(rev_b))):
        da = rev_a[i] if i < len(rev_a) else 1
        db = rev_b[i] if i < len(rev_b) else 1
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            return None
    return tuple(reversed(out))


def _static_shape(t: TensorType) -> tuple[int, ...]:
    """
    Materialise the static part of a (possibly symbolic) shape. Symbolic dims
    are returned as ``-1`` so we can still do shape-equality checks for
    mixed static/symbolic IR.
    """
    out: list[int] = []
    for d in t.shape.data:
        v = getattr(d, "data", None)
        out.append(v if isinstance(v, int) else -1)
    return tuple(out)


# ---------------------------------------------------------------------------
# Gemm op
# ---------------------------------------------------------------------------


@irdl_op_definition
class GemmOp(IRDLOperation):
    """
    `forge.tensor.gemm` — general matrix-multiply with explicit transpose
    flags. Inputs are two tensors of compatible shapes:

        A : !forge.tensor<[..., M, K]>
        B : !forge.tensor<[..., K, N]>
        out : !forge.tensor<[..., M, N]>

    Unlike `forge.graph.add`, gemm does NOT broadcast — the inner K dim
    must match exactly, no implicit dim-1 expansion. The ``transpose_a`` /
    ``transpose_b`` flags only swap M/K (resp. K/N); they do not introduce
    broadcasting.

    Verifier:
        * A.dtype == B.dtype == out.dtype
        * A.shape[-1] == B.shape[-2]  (inner K dim matches)
        * out.shape == (A.shape[:-1], B.shape[-1])  (batch + N)
    """

    name = "forge.tensor.gemm"
    transpose_a = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    transpose_b = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    a = operand_def(TensorType)
    b = operand_def(TensorType)
    out = result_def(TensorType)

    traits = traits_def(Pure(), NoMemoryEffect())

    # Properties are routed through attr-dict so the optional transpose
    # flags can be elided entirely when they take their default values.
    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = "$a `,` $b attr-dict `:` `(` type($a) `,` type($b) `)` `->` type($out)"

    def __init__(
        self,
        a,
        b,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ):
        super().__init__(
            operands=[a, b],
            properties={
                "transpose_a": BoolAttr.from_bool(transpose_a),
                "transpose_b": BoolAttr.from_bool(transpose_b),
            },
        )

    def verify_(self) -> None:
        a_type = _check_tensor(self.a, "forge.tensor.gemm operand 'a'")
        b_type = _check_tensor(self.b, "forge.tensor.gemm operand 'b'")
        out_type = _check_tensor(self.out, "forge.tensor.gemm result")
        if a_type.dtype != b_type.dtype:
            raise VerifyException(
                f"forge.tensor.gemm operand dtypes must match: {a_type.dtype} vs {b_type.dtype}"
            )
        if a_type.dtype != out_type.dtype:
            raise VerifyException(
                "forge.tensor.gemm result dtype must match operand: "
                f"{out_type.dtype} vs {a_type.dtype}"
            )
        a_shape = _static_shape(a_type)
        b_shape = _static_shape(b_type)
        out_shape = _static_shape(out_type)
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise VerifyException(
                "forge.tensor.gemm requires operands with rank >= 2, got "
                f"a.shape={a_shape}, b.shape={b_shape}"
            )
        if a_shape[-1] != b_shape[-2]:
            raise VerifyException(
                "gemm inner dims mismatch: "
                f"a.shape[-1]={a_shape[-1]} != b.shape[-2]={b_shape[-2]} "
                f"(a.shape={a_shape}, b.shape={b_shape})"
            )
        expected_out_shape = (*a_shape[:-1], b_shape[-1])
        if out_shape != expected_out_shape:
            raise VerifyException(
                "forge.tensor.gemm result shape mismatch: "
                f"got {out_shape}, expected {expected_out_shape}"
            )


# ---------------------------------------------------------------------------
# Fused epilogue op
# ---------------------------------------------------------------------------


@irdl_op_definition
class FusedEpilogueOp(IRDLOperation):
    """
    `forge.tensor.fused_epilogue` — applies an optional bias add and an
    activation in a single fused op. Used by the GraphToTensor pass to
    canonicalize ``gemm + bias_add + activation`` sequences.

        x    : !forge.tensor<S>
        bias : !forge.tensor<B>  (optional; if present must broadcast to S)
        out  : !forge.tensor<S>

    The ``activation`` property is required and must be a valid
    ``#forge.activation<...>`` value (``none``, ``relu``, ``gelu``, ``silu``).
    """

    name = "forge.tensor.fused_epilogue"
    activation = prop_def(ActivationAttr)
    x = operand_def(TensorType)
    bias = opt_operand_def(TensorType)
    out = result_def(TensorType)

    traits = traits_def(Pure(), NoMemoryEffect())

    # `activation` is required; route it through attr-dict for MLIR-style
    # `{activation = #forge.activation<gelu>}` syntax.
    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = (
        "$x (`,` $bias^)? attr-dict `:` `(` type($x) (`,` type($bias)^)? `)` `->` type($out)"
    )

    def __init__(
        self,
        x,
        bias=None,
        activation: ActivationAttr = ActivationAttr(Activation.NONE),
    ):
        operands = [x]
        if bias is not None:
            operands.append(bias)
        super().__init__(
            operands=operands,
            properties={"activation": activation},
        )

    def verify_(self) -> None:
        x_type = _check_tensor(self.x, "forge.tensor.fused_epilogue operand 'x'")
        out_type = _check_tensor(self.out, "forge.tensor.fused_epilogue result")
        if x_type.dtype != out_type.dtype:
            raise VerifyException(
                "forge.tensor.fused_epilogue result dtype must match input: "
                f"{out_type.dtype} vs {x_type.dtype}"
            )
        if _static_shape(out_type) != _static_shape(x_type):
            raise VerifyException(
                "forge.tensor.fused_epilogue result shape must equal input "
                f"shape: got {_static_shape(out_type)}, "
                f"expected {_static_shape(x_type)}"
            )
        if self.bias is not None:
            bias_type = _check_tensor(self.bias, "forge.tensor.fused_epilogue bias")
            if bias_type.dtype != x_type.dtype:
                raise VerifyException(
                    "forge.tensor.fused_epilogue bias dtype must match input: "
                    f"{bias_type.dtype} vs {x_type.dtype}"
                )
            if _broadcast_shape(_static_shape(bias_type), _static_shape(x_type)) != _static_shape(
                x_type
            ):
                raise VerifyException(
                    "forge.tensor.fused_epilogue bias shape "
                    f"{_static_shape(bias_type)} is not broadcastable to "
                    f"input shape {_static_shape(x_type)}"
                )


# ---------------------------------------------------------------------------
# Broadcast add op
# ---------------------------------------------------------------------------


@irdl_op_definition
class BroadcastAddOp(IRDLOperation):
    """
    `forge.tensor.broadcast_add` — element-wise addition with NumPy-style
    broadcasting. No autograd semantics; this is the "tensor-level" view of
    what `forge.graph.add` lowers to.

        x    : !forge.tensor<Sx>
        y    : !forge.tensor<Sy>
        out  : !forge.tensor<broadcast(Sx, Sy)>
    """

    name = "forge.tensor.broadcast_add"
    x = operand_def(TensorType)
    y = operand_def(TensorType)
    out = result_def(TensorType)

    traits = traits_def(Pure(), NoMemoryEffect())

    assembly_format = "$x `,` $y attr-dict `:` `(` type($x) `,` type($y) `)` `->` type($out)"

    def __init__(self, x, y):
        super().__init__(operands=[x, y])

    def verify_(self) -> None:
        x_type = _check_tensor(self.x, "forge.tensor.broadcast_add operand 'x'")
        y_type = _check_tensor(self.y, "forge.tensor.broadcast_add operand 'y'")
        out_type = _check_tensor(self.out, "forge.tensor.broadcast_add result")
        if x_type.dtype != y_type.dtype:
            raise VerifyException(
                "forge.tensor.broadcast_add operand dtypes must match: "
                f"{x_type.dtype} vs {y_type.dtype}"
            )
        if out_type.dtype != x_type.dtype:
            raise VerifyException(
                "forge.tensor.broadcast_add result dtype must match operand: "
                f"{out_type.dtype} vs {x_type.dtype}"
            )
        expected_out = _broadcast_shape(_static_shape(x_type), _static_shape(y_type))
        if expected_out is None:
            raise VerifyException(
                "forge.tensor.broadcast_add operand shapes are not "
                f"broadcastable: {_static_shape(x_type)} vs "
                f"{_static_shape(y_type)}"
            )
        if _static_shape(out_type) != expected_out:
            raise VerifyException(
                f"forge.tensor.broadcast_add result shape mismatch: got "
                f"{_static_shape(out_type)}, expected {expected_out}"
            )


# ---------------------------------------------------------------------------
# GELU op
# ---------------------------------------------------------------------------


@irdl_op_definition
class GeluOp(IRDLOperation):
    """
    `forge.tensor.gelu` — Gaussian Error Linear Unit. The optional
    ``approximate`` flag selects the tanh approximation used by PyTorch /
    ONNX; absent (or false) means the exact erf form.

        x   : !forge.tensor<S>
        out : !forge.tensor<S>
    """

    name = "forge.tensor.gelu"
    approximate = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    x = operand_def(TensorType)
    out = result_def(TensorType)

    traits = traits_def(Pure(), NoMemoryEffect())

    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = "$x attr-dict `:` `(` type($x) `)` `->` type($out)"

    def __init__(self, x, approximate: bool = False):
        super().__init__(
            operands=[x],
            properties={"approximate": BoolAttr.from_bool(approximate)},
        )

    def verify_(self) -> None:
        x_type = _check_tensor(self.x, "forge.tensor.gelu operand 'x'")
        out_type = _check_tensor(self.out, "forge.tensor.gelu result")
        if x_type.dtype != out_type.dtype:
            raise VerifyException(
                "forge.tensor.gelu result dtype must match input: "
                f"{out_type.dtype} vs {x_type.dtype}"
            )
        if _static_shape(x_type) != _static_shape(out_type):
            raise VerifyException(
                "forge.tensor.gelu result shape must equal input shape: "
                f"got {_static_shape(out_type)}, "
                f"expected {_static_shape(x_type)}"
            )


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


TensorDialect = Dialect(
    "forge.tensor",
    [
        GemmOp,
        FusedEpilogueOp,
        BroadcastAddOp,
        GeluOp,
    ],
    [
        ActivationAttr,
    ],
)
Tensor = TensorDialect
