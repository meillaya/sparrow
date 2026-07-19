"""ForgeStack FX importer pytest tests (Task 8)."""

from __future__ import annotations

import io

import pytest
import torch
from forge_torch import op_mapping as om
from forge_torch.conftest import FusedMLP
from forge_torch.fx_importer import FxImporter
from torch.fx.experimental.proxy_tensor import make_fx
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer


def _module_roundtrip(mlir_text: str) -> str:
    """Parse the importer's MLIR, verify, and re-print — proves the textual
    output is well-formed forge.graph dialect IR.

    Mirrors ``forge-opt --action=roundtrip`` via the in-process API; we
    discover dialects with the same loader the CLI uses so the load set
    matches what real users get.
    """
    ctx = _dialect_ctx()
    parser = Parser(ctx, mlir_text, "<test>")
    module = parser.parse_module()
    assert isinstance(module, ModuleOp)
    module.verify()
    out = io.StringIO()
    Printer(stream=out).print_op(module)
    return out.getvalue()


def _dialect_ctx() -> Context:
    """xDSL ``Context`` with builtin + func + forge dialects, matching
    ``forge-opt``'s auto-discovery path."""
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func as FuncDialect

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(FuncDialect)
    ctx.load_dialect(_forge_graph_dialect())
    ctx.load_dialect(_forge_tensor_dialect())
    return ctx


def _forge_graph_dialect():
    import importlib

    return importlib.import_module("compiler.lib.Dialect.Graph").Graph


def _forge_tensor_dialect():
    import importlib

    return importlib.import_module("compiler.lib.Dialect.Tensor").Tensor


def test_fx_importer_fused_mlp_roundtrip(fused_mlp_inputs):
    """3-op FusedMLP FX → MLIR roundtrips through xDSL and contains the three
    ``forge.graph`` ops exactly once each."""
    x, w, b = fused_mlp_inputs
    gm = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    importer = FxImporter([x, w, b])
    mlir = importer.import_graph(gm)

    assert mlir.count("forge.graph.matmul") == 1
    assert mlir.count("forge.graph.add") == 1
    assert mlir.count("forge.graph.gelu") == 1

    rounded = _module_roundtrip(mlir)
    assert "forge.graph.matmul" in rounded
    assert "forge.graph.add" in rounded
    assert "forge.graph.gelu" in rounded
    assert rounded.count("forge.graph.matmul") == 1
    assert rounded.count("forge.graph.add") == 1
    assert rounded.count("forge.graph.gelu") == 1


def test_op_mapping_dtype_table():
    """Every common PyTorch dtype maps to the expected forge.graph spelling."""
    assert om.DTYPE_TO_FORGE[torch.float32] == "f32"
    assert om.DTYPE_TO_FORGE[torch.float16] == "f16"
    assert om.DTYPE_TO_FORGE[torch.bfloat16] == "bf16"
    assert om.DTYPE_TO_FORGE[torch.int32] == "i32"
    assert om.DTYPE_TO_FORGE[torch.int8] == "i8"
    assert len(om.DTYPE_TO_FORGE) == 5


def test_op_mapping_unsupported_aten_op_raises():
    """Encountering ``aten.relu`` produces a NotImplementedError naming the op."""

    class HasRelu(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    gm = make_fx(HasRelu(), tracing_mode="fake")(torch.randn(2, 3))
    importer = FxImporter([torch.randn(2, 3)])
    with pytest.raises(NotImplementedError, match="relu"):
        importer.import_graph(gm)


def test_compile_callable_returns_eager_result(fused_mlp_inputs):
    """`compile(FusedMLP(), inputs)(*inputs)` returns a tensor that agrees
    with PyTorch eager up to fp32 atol=1e-5 — Phase 1 still delegates
    through CompiledFusedMLP.__call__ which runs the captured model."""
    from forge_torch.backend import BACKEND, compile

    assert BACKEND == "forgestack"
    x, w, b = fused_mlp_inputs

    compiled = compile(FusedMLP(), [x, w, b])
    got = compiled(x, w, b)

    eager = FusedMLP()(x, w, b)
    assert torch.allclose(got, eager, atol=1e-5)
    assert got.shape == eager.shape
    assert got.dtype == eager.dtype
