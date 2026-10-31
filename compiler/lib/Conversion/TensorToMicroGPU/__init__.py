"""TensorToMicroGPU: lower small tensor graphs to MicroGPU bytecode.

Supported patterns (deliberately loop-free until the ISA grows control
flow):

* one elementwise ``broadcast_add``, ``mul``, or ``relu`` on up to 16
  elements;
* one 4x4 matmul (``forge.tensor.gemm``), which maps to the `MATMUL`
  instruction.

Reductions stay unsupported. The output is a :class:`MicroGpuProgram`
with the instruction words, buffer bindings, and a memory image size;
:mod:`forge_torch.microgpu` wraps it in an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import IntAttr, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp

from backends.microgpu.isa import OpCode, encode
from compiler.lib.Dialect.Graph import TensorType
from compiler.lib.Dialect.Tensor import BroadcastAddOp, GeluOp, GemmOp, MulOp, ReluOp

FLOAT_BYTES = 4
MAX_ELEMENTWISE = 16


class UnsupportedMicroGpuError(ValueError):
    """The graph is outside the loop-free MicroGPU lowering."""


@dataclass(frozen=True)
class MicroGpuBuffer:
    name: str
    role: str
    offset: int
    byte_size: int
    shape: tuple[int, ...]
    input_index: int | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "offset": self.offset,
            "byte_size": self.byte_size,
            "shape": list(self.shape),
            "dtype": "f32",
        }
        if self.input_index is not None:
            payload["input_index"] = self.input_index
        return payload


@dataclass(frozen=True)
class MicroGpuProgram:
    words: tuple[int, ...]
    buffers: tuple[MicroGpuBuffer, ...]
    output: MicroGpuBuffer
    memory_bytes: int

    def header(self) -> dict[str, Any]:
        return {
            "version": 2,
            "entry_point": "microgpu.bytecode.v2",
            "memory_bytes": self.memory_bytes,
            "words": list(self.words),
            "buffers": [buffer.to_json() for buffer in self.buffers],
            "output": self.output.to_json(),
            "output_in_registers": True,
        }

    def to_binary(self) -> bytes:
        import json

        return json.dumps(self.header(), sort_keys=True, separators=(",", ":")).encode()


def _shape(value) -> tuple[int, ...]:
    if not isinstance(value.type, TensorType):
        raise UnsupportedMicroGpuError(f"expected !forge.tensor, got {value.type}")
    dims: list[int] = []
    for dim in value.type.get_shape():
        if not isinstance(dim, IntAttr):
            raise UnsupportedMicroGpuError("MicroGPU lowering requires static shapes")
        dims.append(int(dim.data))
    return tuple(dims)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _args(module: ModuleOp):
    functions = [op for op in module.ops if isinstance(op, FuncOp)]
    if len(functions) != 1:
        raise UnsupportedMicroGpuError("MicroGPU lowering needs exactly one func.func")
    block = functions[0].body.blocks.first
    if block is None:
        raise UnsupportedMicroGpuError("MicroGPU lowering needs a function body")
    return block, [op for op in block.ops if not isinstance(op, ReturnOp)]


def _layout(count: int, inputs: int) -> tuple[list[MicroGpuBuffer], int]:
    buffers: list[MicroGpuBuffer] = []
    cursor = 0
    for index in range(inputs):
        buffers.append(
            MicroGpuBuffer(
                name=("x" if index == 0 else "w" if index == 1 else f"in{index}"),
                role="input",
                offset=cursor,
                byte_size=count * FLOAT_BYTES,
                shape=(count,),
                input_index=index,
            )
        )
        cursor += count * FLOAT_BYTES
    out = MicroGpuBuffer("y", "output", cursor, count * FLOAT_BYTES, (count,))
    return [*buffers, out], cursor + count * FLOAT_BYTES


def _load(register: int, word_offset: int) -> int:
    return encode(OpCode.LOAD_GLOBAL, register, 0, imm=word_offset)


def _store(register: int, word_offset: int) -> int:
    return encode(OpCode.STORE_GLOBAL, 0, register, imm=word_offset)


def _lower_elementwise(op, block) -> MicroGpuProgram:
    if isinstance(op, BroadcastAddOp):
        a, b, result = op.x, op.y, op.out
        compute = OpCode.FADD
    elif isinstance(op, MulOp):
        a, b, result = op.x, op.y, op.out
        compute = OpCode.FMUL
    elif isinstance(op, ReluOp):
        a, b, result = op.x, None, op.out
        compute = OpCode.FRELU
    else:  # pragma: no cover - caller filters
        raise UnsupportedMicroGpuError(f"unsupported elementwise op {op.name}")
    if _shape(a) != _shape(result):
        raise UnsupportedMicroGpuError("elementwise operands must have equal shapes")
    count = _numel(_shape(result))
    if count > MAX_ELEMENTWISE:
        raise UnsupportedMicroGpuError(
            f"loop-free MicroGPU elementwise path supports <= {MAX_ELEMENTWISE} elements"
        )
    input_count = 1 if b is None else 2
    buffers, memory_bytes = _layout(count, input_count)
    offsets = {buffer.name: buffer.offset for buffer in buffers}
    words: list[int] = []
    out_word = offsets["y"] // FLOAT_BYTES
    for index in range(count):
        words.append(_load(0, offsets["x"] // FLOAT_BYTES + index))
        if b is not None:
            words.append(_load(1, offsets["w"] // FLOAT_BYTES + index))
            words.append(encode(compute, 0, 1, imm=0))
        else:
            words.append(encode(compute, 0, 0, imm=0))
        words.append(_store(0, out_word + index))
    for index in range(count):
        words.append(_load(index, out_word + index))
    words.append(encode(OpCode.RETURN, 0, 0, imm=0))
    return MicroGpuProgram(
        words=tuple(words),
        buffers=tuple(buffers),
        output=buffers[-1],
        memory_bytes=memory_bytes,
    )


def _lower_gemm(op: GemmOp) -> MicroGpuProgram:
    a_shape = _shape(op.a)
    b_shape = _shape(op.b)
    out_shape = _shape(op.out)
    if a_shape != (4, 4) or b_shape != (4, 4) or out_shape != (4, 4):
        raise UnsupportedMicroGpuError(
            f"MicroGPU matmul path supports 4x4x4 only, got {a_shape}, {b_shape}"
        )
    # MATMUL reads A at 0, B at 16 words, writes C at 32 words.
    a = MicroGpuBuffer("x", "input", 0, 16 * FLOAT_BYTES, a_shape, 0)
    b = MicroGpuBuffer("w", "input", 16 * FLOAT_BYTES, 16 * FLOAT_BYTES, b_shape, 1)
    c = MicroGpuBuffer("c", "output", 32 * FLOAT_BYTES, 16 * FLOAT_BYTES, out_shape)
    words = [encode(OpCode.MATMUL, 0, 0, imm=0)]
    for index in range(16):
        words.append(_load(index, 32 + index))
    words.append(encode(OpCode.RETURN, 0, 0, imm=0))
    return MicroGpuProgram(
        words=tuple(words),
        buffers=(a, b, c),
        output=c,
        memory_bytes=48 * FLOAT_BYTES,
    )


def lower_tensor_module(module: ModuleOp) -> MicroGpuProgram:
    """Lower a single-op tensor module to a MicroGPU program."""
    block, ops = _args(module)
    if len(ops) != 1:
        raise UnsupportedMicroGpuError(
            "TensorToMicroGPU lowers one op per module; split the graph first"
        )
    op = ops[0]
    if isinstance(op, GemmOp):
        return _lower_gemm(op)
    if isinstance(op, (BroadcastAddOp, MulOp, ReluOp)):
        return _lower_elementwise(op, block)
    if isinstance(op, GeluOp):
        raise UnsupportedMicroGpuError(
            "GELU needs sigmoid composition plus control flow; not shipped yet"
        )
    raise UnsupportedMicroGpuError(f"no MicroGPU equivalent for {op.name}")


__all__ = (
    "MAX_ELEMENTWISE",
    "MicroGpuBuffer",
    "MicroGpuProgram",
    "UnsupportedMicroGpuError",
    "lower_tensor_module",
)
