"""Sparrow FX-importer symbolic-dim pytest tests (Task 15).

Three tests pin the symbolic-dim path added by Task 15:

1. ``test_symbolic_dim_roundtrip`` — when an FX node carries
   ``tensor_meta['symbolic_dim']`` metadata (PyTorch's ``torch.export``
   convention), the importer emits ``!forge.tensor<?x?xKxf32, [...]>``
   with the named dim K + a ``>= 1`` constraint. Uses ``make_fx`` then
   mutates ``node.meta["tensor_meta"].symbolic_dim`` to simulate the
   ``torch.export`` shape environment without bringing in dynamo.

2. ``test_concrete_dim_unchanged`` — a normal FusedMLP (the Phase-1
   canonical shape) keeps emitting concrete dims, since the new
   symbolic path is metadata-driven and the heuristic is opt-in.

3. ``test_heuristic_b_s_naming`` — when ``FxImporter(symbolic_heuristic=True)``
   is used with a 3D FusedMLP-shaped input that has no symbolic-dim
   metadata, dim 0 becomes ``B`` and dim 1 becomes ``S`` per spec §3's
   "B >= 1, S >= 1" example.
"""

from __future__ import annotations

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from forge_torch.conftest import FusedMLP
from forge_torch.fx_importer import FxImporter


def _imported_mlir(module: torch.fx.GraphModule, inputs, *, heuristic: bool = False) -> str:
    """Helper: import the captured graph and return the textual MLIR."""
    importer = FxImporter(inputs, symbolic_heuristic=heuristic)
    return importer.import_graph(module)


def test_symbolic_dim_roundtrip() -> None:
    """``tensor_meta['symbolic_dim'] = ['B', 'S', 'K']`` round-trips as ``BxSxKxf32``.

    The constraint list carries one ``>= 1`` constraint per non-anonymous
    symbolic dim (K is the named dim; B and S would be the anonymous ``?``
    placeholders in the spec example). Verifies that an FX placeholder
    whose ``tensor_meta.symbolic_dim`` is set flows through the importer
    as ``BxSxKxf32`` with the expected constraint.
    """
    x = torch.randn(2, 512, 4096, dtype=torch.float32)
    w = torch.randn(4096, 11008, dtype=torch.float32)
    b = torch.randn(11008, dtype=torch.float32)

    gm = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    # ``tensor_meta`` is a frozen dataclass; replace it wholesale via a
    # shallow dict that carries the symbolic-dim metadata.
    placeholder_x = next(n for n in gm.graph.nodes if n.op == "placeholder")
    orig_meta = placeholder_x.meta["tensor_meta"]
    placeholder_x.meta["tensor_meta"] = {
        "shape": orig_meta.shape,
        "dtype": orig_meta.dtype,
        "symbolic_dim": ("B", "S", "K"),
    }

    mlir = _imported_mlir(gm, [x, w, b])

    assert "BxSxKxf32" in mlir, f"expected BxSxKxf32 in MLIR, got:\n{mlir}"
    assert 'symbolic_constraint<"K", greater_equal, 1>' in mlir, (
        f"expected named K constraint, got:\n{mlir}"
    )


def test_concrete_dim_unchanged() -> None:
    """A normal FusedMLP emits concrete dims (heuristic off by default)."""
    x = torch.randn(2, 512, 4096, dtype=torch.float32)
    w = torch.randn(4096, 11008, dtype=torch.float32)
    b = torch.randn(11008, dtype=torch.float32)

    gm = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    # No symbolic metadata — default importer must keep concrete dims.
    mlir = _imported_mlir(gm, [x, w, b])

    assert "2x512x4096xf32" in mlir, f"expected concrete 2x512x4096xf32, got:\n{mlir}"
    # And no symbolic_constraint attr should leak in.
    assert "symbolic_constraint" not in mlir


def test_heuristic_b_s_naming() -> None:
    """With ``symbolic_heuristic=True``, a 3D input gets ``B, S, <concrete>`` dims."""
    x = torch.randn(2, 512, 4096, dtype=torch.float32)
    w = torch.randn(4096, 11008, dtype=torch.float32)
    b = torch.randn(11008, dtype=torch.float32)

    gm = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    # No symbolic metadata — the heuristic should drive the names.
    mlir = _imported_mlir(gm, [x, w, b], heuristic=True)

    # First two dims of `x` become B and S, third stays concrete 4096.
    assert "BxSx4096xf32" in mlir, f"expected BxSx4096xf32, got:\n{mlir}"
    # And the constraint list carries ``B >= 1, S >= 1``.
    assert 'symbolic_constraint<"B", greater_equal, 1>' in mlir, (
        f"expected B constraint, got:\n{mlir}"
    )
    assert 'symbolic_constraint<"S", greater_equal, 1>' in mlir, (
        f"expected S constraint, got:\n{mlir}"
    )
