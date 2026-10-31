"""In-process importer and lowering pipeline tests."""

from __future__ import annotations

import torch
from xdsl.dialects.builtin import ModuleOp

from compiler.lib.pipeline import lower_to_tensor
from forge_torch.conftest import FusedMLP
from forge_torch.fx_importer import FxImporter


def _inputs() -> list[torch.Tensor]:
    return [torch.randn(2, 8, 128), torch.randn(128, 256), torch.randn(256)]


def test_import_module_returns_verified_module() -> None:
    gm = torch.fx.symbolic_trace(FusedMLP())
    module = FxImporter(_inputs()).import_module(gm)

    assert isinstance(module, ModuleOp)
    text = str(module)
    assert "forge.graph.matmul" in text
    assert "forge.graph.gelu" in text


def test_lower_to_tensor_removes_graph_ops() -> None:
    gm = torch.fx.symbolic_trace(FusedMLP())
    module = FxImporter(_inputs()).import_module(gm)

    lower_to_tensor(module)
    text = str(module)

    assert "forge.graph." not in text
    assert "forge.tensor.gemm" in text
    assert "forge.tensor.broadcast_add" in text
    assert "forge.tensor.gelu" in text


def test_import_graph_matches_import_module_text() -> None:
    gm = torch.fx.symbolic_trace(FusedMLP())
    importer = FxImporter(_inputs())
    module = importer.import_module(gm)
    text = importer.import_graph(gm)

    assert "forge.graph.matmul" in text
    assert text.strip() == str(module).strip()
