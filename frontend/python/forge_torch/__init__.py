"""ForgeStack Python frontend — ``torch.compile(backend="forgestack")`` hook."""

from __future__ import annotations

from forge_torch.backend import BACKEND, CompiledFusedMLP, compile, compile_fused_mlp
from forge_torch.cache import CacheKey, make_cache_key
from forge_torch.cache_store import DEFAULT_CACHE_ROOT, CacheStore
from forge_torch.report import CompilationReport

__all__ = [
    "BACKEND",
    "DEFAULT_CACHE_ROOT",
    "CacheKey",
    "CacheStore",
    "CompilationReport",
    "CompiledFusedMLP",
    "compile",
    "compile_fused_mlp",
    "make_cache_key",
]
