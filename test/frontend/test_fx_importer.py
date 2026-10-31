"""Sparrow FX importer pytest tests (Task 8)."""

from __future__ import annotations

import io

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer

from forge_torch import op_mapping as om
from forge_torch.conftest import FusedMLP
from forge_torch.fx_importer import FxImporter


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
    """Encountering ``aten.silu`` produces a NotImplementedError naming the op."""

    class HasSilu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    gm = make_fx(HasSilu(), tracing_mode="fake")(torch.randn(2, 3))
    importer = FxImporter([torch.randn(2, 3)])
    with pytest.raises(NotImplementedError, match="silu"):
        importer.import_graph(gm)


def test_compile_callable_returns_eager_result(fused_mlp_inputs):
    """`compile(FusedMLP(), inputs)(*inputs)` returns a tensor that agrees
    with PyTorch eager up to fp32 atol=1e-5 — Phase 1 still delegates
    through CompiledFusedMLP.__call__ which runs the captured model."""
    from forge_torch.backend import BACKEND, compile

    assert BACKEND == "sparrow"
    x, w, b = fused_mlp_inputs

    compiled = compile(FusedMLP(), [x, w, b])
    got = compiled(x, w, b)

    eager = FusedMLP()(x, w, b)
    assert torch.allclose(got, eager, atol=1e-5)
    assert got.shape == eager.shape
    assert got.dtype == eager.dtype


def test_layer_norm_importer():
    """``aten.layer_norm`` (via ``F.layer_norm``) → ``forge.tensor.layer_norm``.

    The test passes weight + bias as positional kwargs (matching PyTorch's
    ``F.layer_norm(x, [shape], weight, bias)`` calling convention) so the
    FX graph captures them as plain tensor placeholders via ``node.kwargs``.
    The ``normalized_shape`` is a constant list to avoid ``x.size(-1)`` /
    ``getattr`` traces that the Phase-1 importer doesn't resolve.
    """

    class HasLayerNorm(torch.nn.Module):
        def forward(self, x, w, b):
            return torch.nn.functional.layer_norm(x, [64], w, b)

    x = torch.randn(2, 16, 64)
    w = torch.ones(64)
    b = torch.zeros(64)
    gm = torch.fx.symbolic_trace(HasLayerNorm())
    mlir = FxImporter([x, w, b]).import_graph(gm)

    assert "forge.tensor.layer_norm" in mlir
    assert "forge.graph.layer_norm" not in mlir

    rounded = _module_roundtrip(mlir)
    assert "forge.tensor.layer_norm" in rounded


def test_softmax_importer():
    """``aten.softmax`` (via ``F.softmax``) → ``forge.tensor.softmax``.

    The dim arg is passed as a kwarg (matching PyTorch's
    ``F.softmax(x, dim=-1)`` signature); the importer pulls it out of
    ``node.kwargs['dim']`` via :func:`_extract_softmax_dim`.
    """

    class HasSoftmax(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.softmax(x, dim=-1)

    x = torch.randn(2, 16, 32)
    gm = torch.fx.symbolic_trace(HasSoftmax())
    mlir = FxImporter([x]).import_graph(gm)

    assert "forge.tensor.softmax" in mlir
    assert "forge.graph.softmax" not in mlir

    rounded = _module_roundtrip(mlir)
    assert "forge.tensor.softmax" in rounded


def test_sdpa_importer():
    """``aten.scaled_dot_product_attention`` → ``forge.tensor.scaled_dot_product_attention``.

    The Python wrapper ``F.scaled_dot_product_attention(q, k, v)`` captures
    as ``call_function <built-in function scaled_dot_product_attention>``
    in the FX graph; ``op_mapping.normalize_target`` reduces this to
    ``aten.scaled_dot_product_attention``. The importer emits a single
    fused op (no ``aten.bmm``/``aten.transpose`` decomposition) so the
    downstream codegen can run the fused attention kernel.
    """

    class HasSDPA(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q = torch.randn(2, 16, 64)
    k = torch.randn(2, 16, 64)
    v = torch.randn(2, 16, 64)
    gm = torch.fx.symbolic_trace(HasSDPA())
    mlir = FxImporter([q, k, v]).import_graph(gm)

    assert "forge.tensor.scaled_dot_product_attention" in mlir
    assert "forge.graph.scaled_dot_product_attention" not in mlir

    rounded = _module_roundtrip(mlir)
    assert "forge.tensor.scaled_dot_product_attention" in rounded


def test_elementwise_and_movement_importer():
    """relu / mul / permute are emitted as tensor-dialect ops."""

    class Elementwise(torch.nn.Module):
        def forward(self, x, scale):
            return torch.relu(x * scale).permute(0, 2, 1)

    x = torch.randn(2, 3, 4)
    scale = torch.randn(4)
    gm = torch.fx.symbolic_trace(Elementwise())
    mlir = FxImporter([x, scale]).import_graph(gm)

    assert "forge.tensor.mul" in mlir
    assert "forge.tensor.relu" in mlir
    assert "forge.tensor.transpose" in mlir
    rounded = _module_roundtrip(mlir)
    assert "forge.tensor.transpose" in rounded


def test_addmm_and_bmm_importer():
    """aten.addmm and aten.bmm both lower to gemm-shaped IR."""

    class Addmm(torch.nn.Module):
        def forward(self, bias, a, b):
            return torch.addmm(bias, a, b)

    class Bmm(torch.nn.Module):
        def forward(self, a, b):
            return torch.bmm(a, b)

    bias, a, b = torch.randn(5), torch.randn(4, 8), torch.randn(8, 5)
    mlir = FxImporter([bias, a, b]).import_graph(torch.fx.symbolic_trace(Addmm()))
    assert "forge.tensor.gemm" in mlir
    assert "forge.tensor.broadcast_add" in mlir

    a3, b3 = torch.randn(2, 4, 8), torch.randn(2, 8, 5)
    mlir = FxImporter([a3, b3]).import_graph(torch.fx.symbolic_trace(Bmm()))
    assert "forge.graph.matmul" in mlir
    assert "!forge.tensor<2x4x5xf32>" in mlir


def test_get_attr_parameters_are_lifted():
    """``get_attr`` parameter tensors become extra function arguments."""

    class AttrLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight, self.bias)

    x = torch.randn(2, 8)
    gm = torch.fx.symbolic_trace(AttrLinear())
    importer = FxImporter([x])
    mlir = importer.import_graph(gm)

    assert len(importer.lifted_tensors) == 2
    assert importer.lifted_tensors[0].shape == (4, 8)
    assert importer.lifted_tensors[1].shape == (4,)
    assert "forge.tensor.gemm" in mlir
    assert "forge.tensor.broadcast_add" in mlir


def test_call_module_linear_and_layer_norm():
    """Normal nn.Linear / nn.LayerNorm modules are lowered, not skipped."""

    class Modules(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 4)
            self.ln = torch.nn.LayerNorm(4)

        def forward(self, x):
            return self.ln(self.fc(x))

    x = torch.randn(2, 8)
    gm = torch.fx.symbolic_trace(Modules())
    importer = FxImporter([x])
    mlir = importer.import_graph(gm)

    assert len(importer.lifted_tensors) == 4
    assert "forge.tensor.gemm" in mlir
    assert "forge.tensor.layer_norm" in mlir
    rounded = _module_roundtrip(mlir)
    assert "forge.tensor.layer_norm" in rounded


def test_shape_arithmetic_folds_to_concrete_dims():
    """``B, S, D = x.shape`` is resolved from the example inputs."""

    class ShapeReshape(torch.nn.Module):
        def forward(self, x):
            b, s, d = x.shape
            return x.reshape(b, s * d)

    x = torch.randn(2, 8, 4)
    mlir = FxImporter([x]).import_graph(torch.fx.symbolic_trace(ShapeReshape()))
    assert "!forge.tensor<2x32xf32>" in mlir
    assert "getitem" not in mlir


def test_addmm_rejects_non_default_scalars():
    """Ignoring alpha/beta would silently produce wrong numbers, so reject."""

    class ScaledAddmm(torch.nn.Module):
        def forward(self, bias, a, b):
            return torch.addmm(bias, a, b, beta=0.5)

    with pytest.raises(NotImplementedError, match="beta"):
        FxImporter([torch.randn(5), torch.randn(4, 8), torch.randn(8, 5)]).import_graph(
            torch.fx.symbolic_trace(ScaledAddmm())
        )


def test_make_fx_linear_lifts_get_attr_constants():
    """make_fx stores parameters as ``get_attr`` constants; the importer lifts them."""
    model = torch.nn.Linear(8, 4)
    x = torch.randn(2, 8)
    gm = make_fx(model)(x)
    importer = FxImporter([x])
    mlir = importer.import_graph(gm)

    assert len(importer.lifted_tensors) == 2
    assert "forge.tensor.gemm" in mlir
    assert "forge.tensor.broadcast_add" in mlir
