"""MicroGPU v2: 32-bit words, matmul, and TensorToMicroGPU (M6)."""

from __future__ import annotations

import struct

import pytest
import torch

from backends.microgpu.interp import run
from backends.microgpu.isa import OpCode, decode
from compiler.lib.Conversion.TensorToMicroGPU import (
    UnsupportedMicroGpuError,
    lower_tensor_module,
)
from compiler.lib.pipeline import lower_to_tensor
from forge_torch.fx_importer import FxImporter


def _program(model, inputs):
    gm = torch.fx.symbolic_trace(model)
    module = lower_to_tensor(FxImporter(inputs).import_module(gm))
    return lower_tensor_module(module)


def _run(model, inputs):
    program = _program(model, inputs)
    memory = bytearray(program.memory_bytes)
    input_buffers = [buffer for buffer in program.buffers if buffer.role == "input"]
    for buffer, tensor in zip(input_buffers, inputs, strict=True):
        raw = tensor.detach().numpy().astype("float32").tobytes()
        memory[buffer.offset : buffer.offset + buffer.byte_size] = raw
    state = run(program.words, memory=memory)
    count = program.output.byte_size // 4
    # Elementwise programs leave results in r0..rN and also in memory;
    # the GEMM program leaves them in r0..r15.
    values = [
        struct.unpack("<f", struct.pack("<I", state.registers[i] & 0xFFFFFFFF))[0]
        for i in range(count)
    ]
    return torch.tensor(values).reshape(program.output.shape)


class _Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b


class _Mul(torch.nn.Module):
    def forward(self, a, b):
        return a * b


class _Relu(torch.nn.Module):
    def forward(self, a):
        return torch.relu(a)


class _Gemm(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


@pytest.mark.parametrize(("model", "op"), [(_Add(), torch.add), (_Mul(), torch.mul)])
def test_elementwise_program_matches_eager(model, op) -> None:
    a = torch.randn(4)
    b = torch.randn(4)
    actual = _run(model, [a, b])
    assert torch.allclose(actual, op(a, b), atol=1e-6, rtol=1e-6)


def test_relu_program_matches_eager() -> None:
    a = torch.tensor([-2.0, -0.5, 0.0, 3.0])
    actual = _run(_Relu(), [a])
    assert torch.allclose(actual, torch.relu(a))


def test_four_by_four_gemm_matches_eager() -> None:
    torch.manual_seed(0)
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    program = _program(_Gemm(), [a, b])
    assert decode(program.words[0]).opcode is OpCode.MATMUL
    # The result load addresses exceed the old 16-word field.
    assert max(decode(word).imm for word in program.words[1:]) >= 32

    actual = _run(_Gemm(), [a, b])
    assert torch.allclose(actual, a @ b, atol=1e-5, rtol=1e-5)


class _Sum(torch.nn.Module):
    def forward(self, a):
        return a.sum()


def test_reduction_is_rejected() -> None:
    with pytest.raises((UnsupportedMicroGpuError, NotImplementedError, KeyError)):
        _program(_Sum(), [torch.randn(4)])
