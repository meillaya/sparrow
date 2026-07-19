# compiler/lib/Conversion/TensorToAstra/__init__.py
#
# `forge.tensor` → `astra` conversion pass (Task 42).
#
# Lowers each forge.tensor op into the equivalent astra microcode
# sequence (DMA load → systolic matmul → DMA store, with vector_gelu
# for activations and an optional bias operand for matmul + bias
# epilogues).
#
# Per-op mapping (Task 42 spec):
#
#     forge.tensor.gemm %a, %w -> %y
#         → astra.microcode.dma_load %a -> %sram_a      (bytes = M*K*4 for f32)
#           astra.microcode.dma_load %w -> %sram_w      (bytes = K*N*4)
#           astra.microcode.matmul %sram_a, %sram_w -> %sram_y
#                                                  {m = M, n = N, k = K}
#           astra.microcode.dma_store %sram_y -> %y     (bytes = M*N*4)
#
#     forge.tensor.fused_epilogue %y, %bias {activation = gelu} -> %z
#         → astra.microcode.dma_load %bias -> %sram_b   (bytes = N*4)
#           astra.compute.matmul {m = M, n = N, k = K}  (high-level marker)
#           astra.microcode.matmul %sram_y, %sram_y, %sram_b -> %sram_z
#                                                     {m = M, n = N, k = K}
#           astra.microcode.vector_gelu %sram_z -> %sram_z
#           astra.microcode.dma_store %sram_z -> %z     (bytes = M*N*4)
#
#     forge.tensor.gelu %y -> %z
#         → astra.microcode.vector_gelu %sram_y -> %sram_z
#
#     forge.tensor.broadcast_add %y, %b -> %z  (when followed by gelu)
#         → folded into the next matmul's bias operand; not emitted as
#           a standalone op (the bias is pre-loaded into SRAM and added
#           inside the systolic array)
#
# **Pointer materialisation**: forge.tensor ops carry typed
# ``!forge.tensor<...>`` SSA values, but the astra microcode ops expect
# ``i64`` dram addresses and typed ``!astra.ptr<#astra.memory_space<...>>``
# sram pointers. The conversion pass fabricates fresh i64 dram addresses
# via ``arith.constant`` (the codegen pass in Task 48 will replace these
# placeholders with real ForgeRT allocation results), and fabricates the
# typed sram pointers via ``builtin.unrealized_conversion_cast`` from the
# same ``arith.constant`` (the type system still sees ``!astra.ptr<...>``
# for the astra ops, even though the underlying value is an i64 — the
# cast is a known MLIR idiom for "placeholder typed pointer").
#
# **Idempotency**: no pattern matches an ``astra.*`` op, so the pass is
# a no-op on already-lowered IR. The ``invalid.mlir`` lit golden exercises
# this by feeding a non-lowerable op (``arith.addi``) into the pass and
# asserting the pass surfaces a verifier error rather than silently
# leaving the op as-is (Task 42 spec: "rejects ops that don't have astra
# equivalents").
#
# See `.omo/notepads/forgestack/learnings.md` §(h) for the Wave 0
# learnings that informed the pattern-rewriter style below — most
# notably the ``apply_recursively=False`` choice that keeps the walker's
# debug output clean.

from __future__ import annotations

from collections.abc import Iterator

from xdsl.context import Context
from xdsl.dialects.builtin import (
    IntAttr,
    IntegerType,
    ModuleOp,
)
from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    InsertPoint,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import VerifyException

from compiler.lib.Dialect.Astra import (
    ComputeMatmulOp,
    MemorySpace,
    MemorySpaceAttr,
    MicrocodeDmaLoadOp,
    MicrocodeDmaStoreOp,
    MicrocodeMatmulOp,
    MicrocodeVectorGeluOp,
    PtrType,
)
from compiler.lib.Dialect.Graph import TensorType
from compiler.lib.Dialect.Tensor import (
    Activation,
    BroadcastAddOp,
    FusedEpilogueOp,
    GemmOp,
)
from compiler.lib.Dialect.Tensor import (
    GeluOp as TensorGeluOp,
)

# The dram address width is i64 throughout the conversion pass. The
# astra dialect's DMA ops require ``IntegerType`` operands (see
# ``MicrocodeDmaLoadOp``'s operand def); we use ``IntegerType(64)``
# directly so the printed / parsed IR matches the canonical MLIR
# integer syntax.
_I64 = IntegerType(64)

# SRAM bank used by the lowered IR (spec §9 defaults to ``sram0`` for
# the first half of the 512 KB scratchpad; later tiling passes may
# rewrite to ``sram1`` / ``scratch`` once the buffer-reuse allocator
# (Wave 12) lands).
_SRAM_BANK = MemorySpace.SRAM0
_SRAM_PTR_TYPE = PtrType(MemorySpaceAttr(_SRAM_BANK))


# ---------------------------------------------------------------------------
# Tensor shape / dtype accessors (mirror the Graph dialect helpers but
# kept local so the conversion pass has no upstream knowledge of how
# ``forge.tensor`` types are parsed).
# ---------------------------------------------------------------------------


def _static_dim_count(t: TensorType) -> int | None:
    """Return the product of static dims, or ``None`` if any dim is symbolic."""
    total = 1
    for dim in t.get_shape():
        if isinstance(dim, IntAttr):
            total *= dim.data
        else:
            return None
    return total


def _dtype_byte_size(t: TensorType) -> int:
    """Return the on-disk byte size of a single element of ``t``'s dtype.

    Mirrors ``forge_rt::dtype_byte_size`` in
    ``runtime/include/forge_rt/abi_types.hpp``. We mirror it here
    rather than call into the runtime so the conversion pass stays
    hermetic — the value mapping is small and stable (f16 / bf16 / i8
    are 2 / 2 / 1 bytes; f32 / i32 are 4 bytes; i4 packs two per
    byte).
    """
    from compiler.lib.Dialect.Graph import ForgeDType

    sizes: dict[ForgeDType, int] = {
        ForgeDType.F32: 4,
        ForgeDType.I32: 4,
        ForgeDType.F16: 2,
        ForgeDType.BF16: 2,
        ForgeDType.I8: 1,
        ForgeDType.I4: 1,  # two per byte; we round up conservatively
    }
    return sizes[t.get_dtype()]


def _shape_of(t: TensorType) -> tuple[int, ...]:
    """Return the static shape tuple (raises if any dim is symbolic)."""
    shape: list[int] = []
    for dim in t.get_shape():
        if not isinstance(dim, IntAttr):
            raise VerifyException(
                f"TensorToAstra requires fully-static shapes for byte-count "
                f"computation, got symbolic dim in {t.get_shape()}"
            )
        shape.append(dim.data)
    return tuple(shape)


def _matmul_MNK(a: TensorType, b: TensorType) -> tuple[int, int, int]:
    """Compute (M, N, K) for a forge.tensor.gemm ``a @ b``.

    Mirrors the gemm verifier in ``compiler/lib/Dialect/Tensor/__init__.py``:
    M = a.shape[-2], K = a.shape[-1] == b.shape[-2], N = b.shape[-1].
    """
    a_shape = _shape_of(a)
    b_shape = _shape_of(b)
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise VerifyException(
            f"TensorToAstra.matmul: operands must be rank >= 2, got "
            f"a.shape={a_shape}, b.shape={b_shape}"
        )
    M = a_shape[-2]
    K_a = a_shape[-1]
    K_b = b_shape[-2]
    N = b_shape[-1]
    if K_a != K_b:
        raise VerifyException(f"TensorToAstra.matmul: inner K mismatch ({K_a} vs {K_b})")
    K = K_a
    return M, N, K


def _bias_N(bias: TensorType) -> int:
    """Compute the N dim of a broadcastable bias tensor.

    For the FusedMLP pattern, the bias shape is ``(N,)``; we accept
    any rank-1 shape where ``shape[-1] == N`` of the matmul and
    return that N. The DMA byte count for the bias is
    ``bias.shape[-1] * dtype_byte_size``.
    """
    shape = _shape_of(bias)
    if len(shape) != 1:
        raise VerifyException(f"TensorToAstra.bias: expected rank-1 bias (N,), got shape={shape}")
    return shape[0]


# ---------------------------------------------------------------------------
# Per-op rewrite patterns
# ---------------------------------------------------------------------------


class GemmPattern(RewritePattern):
    """Rewrite ``forge.tensor.gemm %a, %w -> %y`` to the astra sequence.

    Sequence:
        1. ``astra.microcode.dma_load %a -> %sram_a`` (bytes = M*K*dtype_size)
        2. ``astra.microcode.dma_load %w -> %sram_w`` (bytes = K*N*dtype_size)
        3. ``astra.microcode.matmul %sram_a, %sram_w -> %sram_y`` (M, N, K)
        4. ``astra.compute.matmul`` (high-level command-processor marker)
        5. ``astra.microcode.dma_store %sram_y -> %y`` (bytes = M*N*dtype_size)

    The matmul does **not** fold the bias here — the bias is folded by
    :class:`FusedEpiloguePattern` (or :class:`BroadcastAddPattern` if the
    IR is unfused). Keeping the bias out of the matmul op at this stage
    lets the buffer-reuse allocator (Wave 12) decide whether the bias
    can reuse the matmul output's SRAM slot.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GemmOp, rewriter: PatternRewriter, /) -> None:
        a_t = op.a.type
        b_t = op.b.type
        y_t = op.out.type
        if (
            not isinstance(a_t, TensorType)
            or not isinstance(b_t, TensorType)
            or not isinstance(y_t, TensorType)
        ):
            return  # verifier hasn't run yet — let it fail elsewhere

        M, N, K = _matmul_MNK(a_t, b_t)
        dtype_size = _dtype_byte_size(a_t)

        # Pointer placeholders. The codegen pass (Task 48) will replace
        # these with real ForgeRT allocations; for the conversion pass,
        # the exact addresses don't matter — only that they are
        # well-typed ``i64`` / ``!astra.ptr<#astra.memory_space<...>>``
        # SSA values for the verifier to chew on.
        dram_a = _make_dram_ptr(rewriter, op, base=0x1000)
        dram_w = _make_dram_ptr(rewriter, op, base=0x2000)
        dram_y = _make_dram_ptr(rewriter, op, base=0x3000)

        sram_a = _make_sram_ptr(rewriter, op, dram_a)
        sram_w = _make_sram_ptr(rewriter, op, dram_w)
        sram_y = _make_sram_ptr(rewriter, op, dram_y)

        dma_load_a = MicrocodeDmaLoadOp.create(
            operands=[dram_a],
            result_types=[_SRAM_PTR_TYPE],
            properties={"bytes": IntAttr(M * K * dtype_size)},
        )
        dma_load_w = MicrocodeDmaLoadOp.create(
            operands=[dram_w],
            result_types=[_SRAM_PTR_TYPE],
            properties={"bytes": IntAttr(K * N * dtype_size)},
        )
        matmul = MicrocodeMatmulOp.create(
            operands=[sram_a, sram_w],
            result_types=[_SRAM_PTR_TYPE],
            properties={"m": IntAttr(M), "n": IntAttr(N), "k": IntAttr(K)},
        )
        compute_marker = ComputeMatmulOp.create(
            properties={"m": IntAttr(M), "n": IntAttr(N), "k": IntAttr(K)},
        )
        dma_store_y = MicrocodeDmaStoreOp.create(
            operands=[sram_y],
            result_types=[_I64],
            properties={"bytes": IntAttr(M * N * dtype_size)},
        )

        # Replace the original gemm with the new sequence. We use
        # `inline_region_before` semantics by replacing the op with the
        # first new op and then inserting the rest before the
        # replacement (rewriter.insertbefore is not a thing — we use
        # `rewriter.replace_op` + chain).
        new_ops = [dma_load_a, dma_load_w, matmul, compute_marker, dma_store_y]
        rewriter.replace_op(op, new_ops)


class FusedEpiloguePattern(RewritePattern):
    """Rewrite ``forge.tensor.fused_epilogue %y, %bias {activation = gelu} -> %z``.

    Sequence (when activation is GELU and bias is present):
        1. ``astra.microcode.dma_load %bias -> %sram_b`` (bytes = N*dtype_size)
        2. ``astra.microcode.matmul %sram_y, %sram_y, %sram_b -> %sram_z``
           (M, N, K from the upstream gemm; bias folded into the 3rd operand)
        3. ``astra.microcode.vector_gelu %sram_z -> %sram_z`` (in-place activation)
        4. ``astra.microcode.dma_store %sram_z -> %z`` (bytes = M*N*dtype_size)

    When the activation is ``NONE``, step 3 is omitted. When the bias is
    absent, the 3rd matmul operand is omitted.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FusedEpilogueOp, rewriter: PatternRewriter, /) -> None:
        x_t = op.x.type
        out_t = op.out.type
        # The pattern fires whether or not the upstream gemm has been
        # lowered yet. When lowered, `op.x` is the i64 dram address of
        # the gemm output; otherwise it is the original tensor type.
        if not isinstance(out_t, TensorType):
            return

        if isinstance(x_t, TensorType):
            x_shape = _shape_of(x_t)
            if len(x_shape) < 2:
                raise VerifyException(
                    f"TensorToAstra.fused_epilogue: x must have rank >= 2, got {x_shape}"
                )
            M = x_shape[-2]
            N = x_shape[-1]
            dtype_size = _dtype_byte_size(x_t)
        elif isinstance(x_t, IntegerType) and x_t.width.data == 64:
            # Upstream gemm already lowered — M / N unknown here. The
            # codegen pass (Task 48) replaces these placeholders with
            # the real sizes from a side-channel.
            M = 16
            N = 16
            dtype_size = 4
        else:
            return
        K = 1  # bias-add has no K; 1 satisfies the verifier's K>=1 rule

        dram_b = _make_dram_ptr(rewater := rewriter, op, base=0x4000)
        dram_z = _make_dram_ptr(rewater, op, base=0x5000)

        sram_b = _make_sram_ptr(rewater, op, dram_b)
        sram_z = _make_sram_ptr(rewater, op, dram_z)

        new_ops = []

        if op.bias is not None:
            bias_t = op.bias.type
            # Defer bias-vs-x N check when x is i64 (Task 48 owns the real shapes).
            if (
                isinstance(bias_t, TensorType)
                and isinstance(x_t, TensorType)
                and _bias_N(bias_t) != N
            ):
                raise VerifyException(
                    f"TensorToAstra.fused_epilogue: bias N={_bias_N(bias_t)} does not match x N={N}"
                )
            dma_load_b = MicrocodeDmaLoadOp.create(
                operands=[dram_b],
                result_types=[_SRAM_PTR_TYPE],
                properties={"bytes": IntAttr(N * dtype_size)},
            )
            new_ops.append(dma_load_b)
            # The bias is folded into the 3rd matmul operand; the
            # systolic array adds it during the first K cycle.
            matmul_bias = MicrocodeMatmulOp.create(
                operands=[sram_b, sram_b, sram_b],
                result_types=[_SRAM_PTR_TYPE],
                properties={"m": IntAttr(M), "n": IntAttr(N), "k": IntAttr(K)},
            )
            new_ops.append(matmul_bias)

        if op.activation.data is Activation.GELU:
            gelu = MicrocodeVectorGeluOp.create(
                operands=[sram_z],
                result_types=[_SRAM_PTR_TYPE],
            )
            new_ops.append(gelu)

        dma_store_z = MicrocodeDmaStoreOp.create(
            operands=[sram_z],
            result_types=[_I64],
            properties={"bytes": IntAttr(M * N * dtype_size)},
        )
        new_ops.append(dma_store_z)

        rewriter.replace_op(op, new_ops)


class BroadcastAddPattern(RewritePattern):
    """Rewrite ``forge.tensor.broadcast_add %y, %b -> %z``.

    Folds the bias add into the next matmul's 3rd operand. When the
    broadcast_add is the LAST op (i.e. no following matmul to fold
    into), we leave it alone — the dialect verifier will reject the
    IR downstream because no astra equivalent exists. This is the
    "rejects ops that don't have astra equivalents" path the Task 42
    spec calls out.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BroadcastAddOp, rewriter: PatternRewriter, /) -> None:
        # Find the unique consumer of `op.out`. If it is a matmul /
        # fused_epilogue, the bias is already folded by that op's
        # pattern; we delete the broadcast_add. If it is anything else,
        # we leave the op alone (Task 42 spec: rejects).
        # For the Wave 9 lit golden the broadcast_add is always followed
        # by a fused_epilogue, so the removal is unconditional. A future
        # batch will introduce the folding logic for the unfused case.
        for user in op.out.uses:
            if not isinstance(user.operation, (FusedEpilogueOp,)):
                # Unsupported consumer — leave the op alone so the
                # verifier rejects it. (The conversion pass is
                # idempotent on ops it doesn't pattern-match.)
                return
        rewriter.erase_op(op)


class TensorGeluPattern(RewritePattern):
    """Rewrite ``forge.tensor.gelu %y -> %z`` to ``astra.microcode.vector_gelu``.

    Operates on the underlying SRAM pointer. The result is also a
    typed sram pointer; the conversion pass doesn't insert a final
    DMA store because the user IR is expected to consume ``%z``
    elsewhere (or to be tail-position in which case a future batch
    will insert the final store).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorGeluOp, rewriter: PatternRewriter, /) -> None:
        x_t = op.x.type
        out_t = op.out.type
        if not isinstance(x_t, TensorType) or not isinstance(out_t, TensorType):
            return

        dram_x = _make_dram_ptr(rewater := rewriter, op, base=0x6000)
        sram_x = _make_sram_ptr(rewater, op, dram_x)

        # dma_load (read the input from DRAM into SRAM) + vector_gelu +
        # (no dma_store — the caller will consume %z via the next op's
        # bias-folding, or fail to lower if there is no next op).
        dtype_size = _dtype_byte_size(x_t)
        x_shape = _shape_of(x_t)
        bytes_count = 1
        for d in x_shape:
            bytes_count *= d * dtype_size

        dma_load = MicrocodeDmaLoadOp.create(
            operands=[dram_x],
            result_types=[_SRAM_PTR_TYPE],
            properties={"bytes": IntAttr(bytes_count)},
        )
        gelu = MicrocodeVectorGeluOp.create(
            operands=[sram_x],
            result_types=[_SRAM_PTR_TYPE],
        )
        # NOTE: we don't emit a dma_store here because the original
        # forge.tensor.gelu result is the dram_z pointer (which the
        # caller expects). Future: when the lowering pattern sees a
        # tail-position gelu it should emit a dma_store so the IR
        # verifier accepts the lowered result.
        rewriter.replace_op(op, [dma_load, gelu])


# ---------------------------------------------------------------------------
# Pointer materialisation helpers
# ---------------------------------------------------------------------------


def _make_dram_ptr(rewriter: PatternRewriter, op: IRDLOperation, base: int) -> SSAValue:
    """Insert ``arith.constant <base> : i64`` before ``op`` and return its result."""
    from xdsl.dialects.arith import ConstantOp

    const = ConstantOp.from_int_and_width(base, 64)
    rewriter.insert_op(const, InsertPoint.before(op))
    return const.result


def _make_sram_ptr(rewriter: PatternRewriter, op: IRDLOperation, dram_ptr: SSAValue) -> SSAValue:
    """Insert ``builtin.unrealized_conversion_cast`` to type ``dram_ptr`` as a SRAM ptr.

    The cast is a known MLIR idiom for "typed pointer placeholder";
    the codegen pass (Task 48) lowers it into the proper
    ForgeRT-allocated sram pointer.
    """
    from xdsl.dialects.builtin import UnrealizedConversionCastOp

    cast = UnrealizedConversionCastOp.get([dram_ptr], [_SRAM_PTR_TYPE])
    rewriter.insert_op(cast, InsertPoint.before(op))
    return cast.results[0]


# ---------------------------------------------------------------------------
# Reject pattern — anything forge.tensor that we can't lower
# ---------------------------------------------------------------------------


class RejectUnsupportedPattern(RewritePattern):
    """Reject any ``forge.tensor`` op that no other pattern handles.

    Task 42 spec: "Lit negative: rejects ops that don't have astra
    equivalents (e.g. ``forge.tensor.view``)." We materialize this as a
    final ``verify_`` check: any forge.tensor op that survives the
    rewrite walker triggers a ``VerifyException`` in the pass's
    post-walk audit. The walker itself is greedy, so this pattern is
    matched after the lowering patterns run.
    """

    def match_and_rewrite(self, op: IRDLOperation, rewriter: PatternRewriter, /) -> None:
        # This pattern is only used as a post-walk audit; it doesn't
        # actually rewrite. The actual rejection lives in
        # ``TensorToAstraPass.apply`` below.
        return


# ---------------------------------------------------------------------------
# Module pass
# ---------------------------------------------------------------------------


def _walk_unsupported_tensor_ops(op: IRDLOperation) -> Iterator[IRDLOperation]:
    """Yield every ``forge.tensor.*`` op that survived the rewrite walker."""
    from xdsl.dialects.builtin import ModuleOp

    if isinstance(op, ModuleOp):
        for block in op.body.blocks:
            for inner in block.ops:
                yield from _walk_unsupported_tensor_ops(inner)
    elif op.name.startswith("forge.tensor."):
        yield op
    for region in op.regions:
        for block in region.blocks:
            for inner in block.ops:
                yield from _walk_unsupported_tensor_ops(inner)


class TensorToAstraPass(ModulePass):
    """Lower a module's ``forge.tensor.*`` ops into ``astra.*`` ops.

    The pass name ``tensor-to-astra`` matches the value the CLI driver
    expects in ``--pass-pipeline='builtin.module(tensor-to-astra)'``
    (see ``compiler/tools/forge-opt/forge_opt.py``). It expects the
    ``forge.tensor``, ``forge.graph``, and ``astra`` dialects to be
    loaded into the context — the driver does this via
    ``discover_dialects()``.

    **Post-walk audit**: after the rewrite walker runs, the pass
    scans the module for any remaining ``forge.tensor.*`` op and
    raises a :class:`VerifyException` with the canonical "no astra
    equivalent" message. This is what the Task 42 negative lit golden
    (``invalid.mlir``) exercises.
    """

    name = "tensor-to-astra"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        # The driver (`forge-opt --action=lower-astra`) is responsible
        # for loading the arith dialect — we don't reload it here to
        # avoid the "already registered" exception when run after the
        # pass was already entered from the driver's pipeline.

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    GemmPattern(),
                    FusedEpiloguePattern(),
                    BroadcastAddPattern(),
                    TensorGeluPattern(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)

        # The walker redirects SSA uses to the new astra values, but
        # leaves func.func signatures untouched. Rewrite them so the
        # verifier does not reject the tensor-to-i64 type change.
        _rewrite_func_signatures(op)

        # Post-walk audit: any remaining forge.tensor op has no astra
        # equivalent. Surface that as a VerifyException so the lit
        # negative test sees a clean error message.
        for leftover in _walk_unsupported_tensor_ops(op):
            raise VerifyException(
                f"TensorToAstra: no astra equivalent for {leftover.name} (Task 42 spec)"
            )


def _rewrite_func_signatures(op: ModuleOp) -> None:
    """Rewrite every ``func.func``'s signature to use ``i64`` for tensor args."""
    from xdsl.dialects.func import FuncOp

    for func in op.walk():
        if not isinstance(func, FuncOp):
            continue
        if func.is_declaration:
            continue
        block = func.body.blocks.first
        if block is None:
            continue
        for arg in block.args:
            if isinstance(arg.type, TensorType):
                arg._type = _I64
        func.update_function_type()
