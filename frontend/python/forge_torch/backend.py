"""ForgeStack ``torch.compile`` backend stub (Task 8, Phase 1).

Spec §3 says ForgeTorch has six jobs:

1. graph capture
2. input metadata extraction
3. FX-to-MLIR import
4. compilation-cache lookup
5. invocation of compiled runtime
6. safe fallback

Phase 1 implements only (1)-(3) with a thin (5) stub that delegates back to
PyTorch eager. The cache lookup (4) lands in Task 13, and (6) needs the FX
partitioning path that ships with Phase 2. ``CompiledFusedMLP.__call__`` is
the eager stub the Phase-1 test suite exercises.
"""

from __future__ import annotations

import torch

from forge_torch.fx_importer import FxImporter

BACKEND = "forgestack"


def compile(model: torch.nn.Module, example_inputs: list[torch.Tensor]) -> CompiledFusedMLP:
    """Compile a PyTorch model using ``torch.compile(backend="forgestack")`` semantics.

    Phase 1 captures the FX graph, runs :class:`FxImporter` to obtain
    ``forge.graph`` MLIR, and returns a :class:`CompiledFusedMLP` that
    delegates to eager. The lowered-execution path lands once ForgeRT
    (Tasks 10-12) and the compilation cache (Task 13) are in place.
    """
    gm = torch.fx.symbolic_trace(model)
    importer = FxImporter(example_inputs)
    mlir_text = importer.import_graph(gm)
    return CompiledFusedMLP(model, example_inputs, mlir_text)


def compile_fused_mlp(
    model: torch.nn.Module, example_inputs: list[torch.Tensor]
) -> CompiledFusedMLP:
    """Convenience: explicit name for the FusedMLP compile path."""
    return compile(model, example_inputs)


class CompiledFusedMLP:
    """Stub: wraps eager execution while the C++ runtime (Tasks 10-12) lands.

    The textual ``mlir_text`` is retained for diagnostics (and for the
    pytest round-trip assertion in ``test_fx_importer.py``); the eventual
    lowered-execution backend will swap ``__call__`` for a call into
    ForgeRT (Task 11).
    """

    def __init__(
        self, model: torch.nn.Module, example_inputs: list[torch.Tensor], mlir_text: str
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.mlir_text = mlir_text

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        return self.model(*args)
