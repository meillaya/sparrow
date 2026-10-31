"""Compile a supported ``forge.tensor`` graph to an Astra v2 artifact.

This is the compiler side of the Astra backend. It walks verified
tensor IR (no regex), lays every buffer out in DRAM and SRAM, tiles K
with accumulation, chunks DMA transfers below the 15-bit immediate, and
serializes the instruction words plus a buffer-binding header into
``ArtifactDescriptor.binary`` for the runtime's binary executor.

Supported shape: one 16x16 gemm with optional rank-1 bias and GELU.
Larger matrices are tiled by the graph partitioner before they reach
this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Final

from xdsl.dialects.builtin import IntAttr, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp

from compiler.lib.Dialect.Astra.microcode import MAX_DMA_BYTES, OpCode, encode
from compiler.lib.Dialect.Graph import TensorType
from compiler.lib.Dialect.Tensor import BroadcastAddOp, GeluOp, GemmOp
from compiler.lib.Dialect.Tensor import FusedEpilogueOp as TensorFusedEpilogueOp

TILE_DIM: Final = 16
TILE_ELEMENTS: Final = TILE_DIM * TILE_DIM
TILE_BYTES: Final = TILE_ELEMENTS * 4
FLOAT_BYTES: Final = 4
_ALIGN: Final = 64
_ENTRY_POINT: Final = "astra.microcode.v2"


class UnsupportedAstraError(ValueError):
    """The graph is outside the single-tile Astra lowering."""


@dataclass(frozen=True)
class AstraBuffer:
    name: str
    role: str  # "input" | "output"
    dram_offset: int
    byte_size: int
    shape: tuple[int, ...]
    input_index: int | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "dram_offset": self.dram_offset,
            "byte_size": self.byte_size,
            "shape": list(self.shape),
            "dtype": "f32",
        }
        if self.input_index is not None:
            payload["input_index"] = self.input_index
        return payload


@dataclass(frozen=True)
class AstraLayout:
    a: AstraBuffer
    b: AstraBuffer
    bias: AstraBuffer | None
    out: AstraBuffer
    sram_acc: int
    sram_stage: int
    sram_b: int
    sram_bias: int
    dram_bytes: int


@dataclass(frozen=True)
class AstraArtifact:
    words: tuple[int, ...]
    layout: AstraLayout
    entry_point: str = _ENTRY_POINT
    version: int = 2

    def header(self) -> dict[str, Any]:
        buffers = [self.layout.a.to_json(), self.layout.b.to_json()]
        if self.layout.bias is not None:
            buffers.append(self.layout.bias.to_json())
        buffers.append(self.layout.out.to_json())
        return {
            "version": self.version,
            "entry_point": self.entry_point,
            "dram_bytes": self.layout.dram_bytes,
            "words": list(self.words),
            "buffers": buffers,
            "output": self.layout.out.to_json(),
        }

    def to_binary(self) -> bytes:
        return json.dumps(self.header(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _align(value: int, alignment: int = _ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def _static_shape(value) -> tuple[int, ...]:
    if not isinstance(value.type, TensorType):
        raise UnsupportedAstraError(f"expected !forge.tensor, got {value.type}")
    dims: list[int] = []
    for dim in value.type.get_shape():
        if not isinstance(dim, IntAttr):
            raise UnsupportedAstraError("Astra lowering requires static shapes")
        dims.append(int(dim.data))
    return tuple(dims)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _f32(value) -> None:
    if not isinstance(value.type, TensorType):
        raise UnsupportedAstraError(f"expected !forge.tensor, got {value.type}")
    if value.type.get_dtype().value != "f32":
        raise UnsupportedAstraError("Astra lowering supports fp32 only")


def _layout(a_shape: tuple[int, ...], b_shape: tuple[int, ...], bias: bool) -> AstraLayout:
    """Lay buffers out so every encoded start address fits 16 bits.

    The instruction word stores byte offsets in 16-bit fields. Output is
    written after the last DMA load, so it can safely alias the input
    region at offset 0; the bias (needed until the epilogue) sits at 0
    and is overwritten by the final store.
    """
    a_bytes = _numel(a_shape) * FLOAT_BYTES
    b_bytes = _numel(b_shape) * FLOAT_BYTES
    bias_bytes = TILE_DIM * FLOAT_BYTES
    out = AstraBuffer("y", "output", 0, TILE_BYTES, (TILE_DIM, TILE_DIM))
    bias_buffer: AstraBuffer | None = None
    cursor = 0
    if bias:
        bias_buffer = AstraBuffer("b", "input", 0, bias_bytes, (TILE_DIM,), 2)
        cursor = bias_bytes
    a_offset = _align(cursor)
    a = AstraBuffer("x", "input", a_offset, a_bytes, a_shape, 0)
    b_offset = _align(a_offset + a_bytes)
    b = AstraBuffer("w", "input", b_offset, b_bytes, b_shape, 1)
    # Only encoded start addresses must fit the 16-bit field; a DMA may
    # cross the 64 KiB boundary because its size travels separately.
    return AstraLayout(
        a=a,
        b=b,
        bias=bias_buffer,
        out=out,
        sram_acc=0,
        sram_stage=TILE_BYTES,
        sram_b=2 * TILE_BYTES,
        sram_bias=3 * TILE_BYTES,
        dram_bytes=max(TILE_BYTES, _align(a_offset + a_bytes + b_bytes + 64)),
    )


def _dma_words(
    opcode: OpCode,
    dst_offset: int,
    src_offset: int,
    byte_count: int,
) -> list[int]:
    """Chunk a DMA transfer so every immediate fits the 15-bit field."""
    if byte_count < 0:
        raise UnsupportedAstraError("negative DMA byte count")
    words: list[int] = []
    remaining = byte_count
    consumed = 0
    while remaining > 0:
        chunk = min(remaining, MAX_DMA_BYTES)
        words.append(encode(opcode, dst_offset + consumed, src_offset + consumed, imm=chunk))
        consumed += chunk
        remaining -= chunk
    return words


def _lower_ops(module: ModuleOp):
    functions = [op for op in module.ops if isinstance(op, FuncOp)]
    if len(functions) != 1:
        raise UnsupportedAstraError("Astra lowering needs exactly one func.func")
    block = functions[0].body.blocks.first
    if block is None:
        raise UnsupportedAstraError("Astra lowering needs a function body")
    ops = [op for op in block.ops if not isinstance(op, ReturnOp)]
    gemm: GemmOp | None = None
    add: BroadcastAddOp | None = None
    gelu: GeluOp | None = None
    fused: TensorFusedEpilogueOp | None = None
    for op in ops:
        if isinstance(op, GemmOp) and gemm is None:
            gemm = op
        elif isinstance(op, BroadcastAddOp) and add is None:
            add = op
        elif isinstance(op, GeluOp) and gelu is None:
            gelu = op
        elif isinstance(op, TensorFusedEpilogueOp) and fused is None:
            fused = op
        else:
            raise UnsupportedAstraError(f"unsupported op in Astra region: {op.name}")
    if gemm is None:
        raise UnsupportedAstraError("Astra region has no forge.tensor.gemm")
    return block, gemm, add, gelu, fused


def build_astra_artifact(module: ModuleOp) -> AstraArtifact:
    """Build an Astra artifact from a lowered single-gemm module."""
    block, gemm, add, gelu, fused = _lower_ops(module)
    _f32(gemm.a)
    _f32(gemm.b)
    a_shape = _static_shape(gemm.a)
    b_shape = _static_shape(gemm.b)
    out_shape = _static_shape(gemm.out)
    if a_shape != (TILE_DIM, a_shape[1]) or b_shape[1] != TILE_DIM:
        raise UnsupportedAstraError(f"Astra v2 lowers 16xK @ Kx16 tiles, got {a_shape} @ {b_shape}")
    if out_shape != (TILE_DIM, TILE_DIM):
        raise UnsupportedAstraError(f"Astra output must be 16x16, got {out_shape}")
    k = a_shape[1]
    if k <= 0:
        raise UnsupportedAstraError("Astra gemm needs K >= 1")

    bias_value = None
    if add is not None:
        if add.x is not gemm.out:
            raise UnsupportedAstraError("Astra bias add must consume the gemm output")
        bias_value = add.y
    elif fused is not None and fused.bias is not None:
        bias_value = fused.bias
    if gelu is not None and gelu.x is not (add.out if add is not None else gemm.out):
        raise UnsupportedAstraError("Astra GELU must consume the epilogue result")
    if fused is not None:
        activation = fused.activation.data.value
        if activation not in {"gelu", "none"}:
            raise UnsupportedAstraError(f"Astra v2 supports gelu/none, got {activation}")

    has_bias = bias_value is not None
    layout = _layout(a_shape, b_shape, has_bias)
    arg_indices = {arg: index for index, arg in enumerate(block.args)}
    if gemm.a not in arg_indices or gemm.b not in arg_indices:
        raise UnsupportedAstraError("Astra inputs must be function arguments")
    layout = replace(
        layout,
        a=replace(layout.a, input_index=arg_indices[gemm.a]),
        b=replace(layout.b, input_index=arg_indices[gemm.b]),
    )
    if layout.bias is not None and bias_value is not None:
        if bias_value not in arg_indices:
            raise UnsupportedAstraError("Astra bias must be a function argument")
        layout = replace(layout, bias=replace(layout.bias, input_index=arg_indices[bias_value]))

    words: list[int] = []
    k_tiles = (k + TILE_DIM - 1) // TILE_DIM
    for tile in range(k_tiles):
        kt = min(TILE_DIM, k - tile * TILE_DIM)
        destination = layout.sram_acc if tile == 0 else layout.sram_stage
        for row in range(TILE_DIM):
            source = layout.a.dram_offset + (row * k + tile * TILE_DIM) * FLOAT_BYTES
            words.extend(
                _dma_words(
                    OpCode.DMA_LOAD,
                    destination + row * kt * FLOAT_BYTES,
                    source,
                    kt * FLOAT_BYTES,
                )
            )
        b_source = layout.b.dram_offset + tile * TILE_DIM * TILE_DIM * FLOAT_BYTES
        words.extend(
            _dma_words(
                OpCode.DMA_LOAD,
                layout.sram_b,
                b_source,
                kt * TILE_DIM * FLOAT_BYTES,
            )
        )
        words.append(encode(OpCode.MATMUL, destination, layout.sram_b, imm=kt))
        if tile > 0:
            words.append(
                encode(OpCode.VECTOR_ADD, layout.sram_acc, layout.sram_stage, imm=TILE_ELEMENTS)
            )

    if layout.bias is not None:
        words.extend(
            _dma_words(
                OpCode.DMA_LOAD,
                layout.sram_bias,
                layout.bias.dram_offset,
                layout.bias.byte_size,
            )
        )
        words.append(
            encode(
                OpCode.VECTOR_ADD_BROADCAST,
                layout.sram_acc,
                layout.sram_bias,
                imm=TILE_ELEMENTS,
            )
        )

    do_gelu = gelu is not None or (fused is not None and fused.activation.data.value == "gelu")
    if do_gelu:
        words.append(
            encode(OpCode.VECTOR_GELU, layout.sram_acc, layout.sram_acc, imm=TILE_ELEMENTS)
        )
    words.extend(_dma_words(OpCode.DMA_STORE, layout.out.dram_offset, layout.sram_acc, TILE_BYTES))
    words.append(encode(OpCode.HALT, 0, 0))
    return AstraArtifact(words=tuple(words), layout=layout)


def layout_offsets(artifact: AstraArtifact) -> dict[str, int]:
    """Return the DRAM offsets the artifact binds, for diagnostics and tests."""
    offsets = {"x": artifact.layout.a.dram_offset, "w": artifact.layout.b.dram_offset}
    if artifact.layout.bias is not None:
        offsets["b"] = artifact.layout.bias.dram_offset
    offsets["y"] = artifact.layout.out.dram_offset
    return offsets


def build_astra_plan(module: ModuleOp) -> object:
    """Wrap an Astra artifact in a version 3 execution plan.

    The plan has one node on an Astra device. Inputs are ordered by the
    header's ``input_index`` so the positional plan binding matches the
    provider's DRAM bindings.
    """
    from forge_torch.plan import F32_DTYPE, BuiltPlan, TensorSpec

    artifact = build_astra_artifact(module)
    header = artifact.header()
    inputs: list[TensorSpec] = []
    for buffer in sorted(
        (item for item in header["buffers"] if item["role"] == "input"),
        key=lambda item: item["input_index"],
    ):
        inputs.append(
            TensorSpec(
                id=len(inputs),
                name=str(buffer["name"]),
                shape=tuple(int(dim) for dim in buffer["shape"]),
                dtype=F32_DTYPE,
            )
        )
    output = TensorSpec(
        id=len(inputs),
        name="y",
        shape=tuple(int(dim) for dim in header["output"]["shape"]),
        dtype=F32_DTYPE,
    )
    tensors = [*inputs, output]
    plan = {
        "version": 3,
        "name": "astra_tile",
        "artifacts": [
            {
                "target": 2,
                "entry_point": artifact.entry_point,
                "binary": list(artifact.to_binary()),
                "arguments": [
                    {
                        "name": spec.name,
                        "dtype": spec.dtype,
                        "is_input": spec.id != output.id,
                        "shape": list(spec.shape),
                    }
                    for spec in tensors
                ],
                "guards": [],
                "workspace_bytes": 0,
                "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
            }
        ],
        "devices": [{"id": 0, "name": "astra", "target": 2, "memory_space": 0}],
        "tensors": [
            {
                "id": spec.id,
                "name": spec.name,
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "memory_space": 0,
            }
            for spec in tensors
        ],
        "allocations": [],
        "nodes": [
            {
                "id": 0,
                "device": 0,
                "artifact": 0,
                "inputs": [spec.id for spec in inputs],
                "outputs": [output.id],
                "dependencies": [],
                "launch": {"grid": [], "block": [], "shared_memory_bytes": 0},
            }
        ],
        "inputs": [{"tensor_id": spec.id, "external_name": spec.name} for spec in inputs],
        "outputs": [{"tensor_id": output.id, "external_name": output.name}],
    }
    return BuiltPlan(plan=plan, inputs=tuple(inputs), outputs=(output,))


__all__ = (
    "TILE_BYTES",
    "TILE_DIM",
    "TILE_ELEMENTS",
    "AstraArtifact",
    "AstraBuffer",
    "AstraLayout",
    "UnsupportedAstraError",
    "build_astra_artifact",
    "build_astra_plan",
    "layout_offsets",
)
