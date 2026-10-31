"""Astra v2: layout, DMA chunking, K tiling, and binary execution (M5)."""

from __future__ import annotations

import importlib
import json

import pytest
import torch
import torch.nn.functional as F

from compiler.lib.pipeline import lower_to_tensor
from forge_torch.astra import (
    TILE_BYTES,
    UnsupportedAstraError,
    _dma_words,
    build_astra_artifact,
    build_astra_plan,
    layout_offsets,
)
from forge_torch.fx_importer import FxImporter

pytest.importorskip("_sparrow_rt")
_sparrow_rt = importlib.import_module("_sparrow_rt")


class _Tile(torch.nn.Module):
    def forward(self, x, w, b):
        return F.gelu(x @ w + b)


def _lowered(model, inputs):
    gm = torch.fx.symbolic_trace(model)
    return lower_to_tensor(FxImporter(inputs).import_module(gm))


@pytest.mark.parametrize("k", [4, 20, 512])
def test_compiler_generated_tiles_match_eager(k: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(16, k)
    w = torch.randn(k, 16)
    b = torch.randn(16)
    model = _Tile()

    plan = build_astra_plan(_lowered(model, [x, w, b]))
    outputs = _sparrow_rt.execute_plan(plan.to_json(), [x.numpy(), w.numpy(), b.numpy()])
    actual = torch.from_numpy(outputs[0])

    assert actual.shape == (16, 16)
    assert torch.allclose(actual, model(x, w, b), atol=1e-4, rtol=1e-4)


def test_artifact_header_binds_buffers_to_dram_offsets() -> None:
    torch.manual_seed(0)
    x, w, b = torch.randn(16, 4), torch.randn(4, 16), torch.randn(16)
    artifact = build_astra_artifact(_lowered(_Tile(), [x, w, b]))
    header = json.loads(artifact.to_binary())

    assert header["version"] == 2
    assert header["entry_point"] == "astra.microcode.v2"
    assert header["output"]["byte_size"] == TILE_BYTES
    offsets = layout_offsets(artifact)
    assert offsets["y"] == 0
    assert set(offsets) == {"x", "w", "b", "y"}
    assert all(0 <= offset <= 0xFFFF for offset in offsets.values())
    roles = {buffer["role"] for buffer in header["buffers"]}
    assert roles == {"input", "output"}
    assert {buffer["input_index"] for buffer in header["buffers"] if buffer["role"] == "input"} == {
        0,
        1,
        2,
    }


def test_dma_chunks_stay_under_the_15_bit_immediate() -> None:
    words = _dma_words(1, 0, 0, 40000)  # OpCode.DMA_LOAD
    sizes = [word & 0x7FFF for word in words]
    starts = [(word >> 16) & 0xFFFF for word in words]
    assert sizes == [0x7FFF, 40000 - 0x7FFF]
    assert starts == [0, 0x7FFF]
    assert all(size < 0x8000 for size in sizes)


def test_non_tile_shape_is_rejected() -> None:
    x, w, b = torch.randn(8, 4), torch.randn(4, 16), torch.randn(16)
    with pytest.raises(UnsupportedAstraError, match="16xK"):
        build_astra_artifact(_lowered(_Tile(), [x, w, b]))
