"""ForgeStack Python frontend — ``torch.compile(backend="forgestack")`` hook."""

from __future__ import annotations

from forge_torch.backend import BACKEND, CompiledFusedMLP, compile, compile_fused_mlp

__all__ = ["BACKEND", "CompiledFusedMLP", "compile", "compile_fused_mlp"]
