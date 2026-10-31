"""Sparrow Python frontend package."""

from __future__ import annotations

from forge_torch.backend import BACKEND, CompiledFusedMLP, compile, compile_fused_mlp
from forge_torch.cache import CacheKey, make_cache_key
from forge_torch.cache_store import DEFAULT_CACHE_ROOT, CacheStore
from forge_torch.dynamic_compile import DynamicCompiledFusedMLP, dynamic_compile
from forge_torch.report import CompilationReport

__all__ = [
    "BACKEND",
    "DEFAULT_CACHE_ROOT",
    "CacheKey",
    "CacheStore",
    "CompilationReport",
    "CompiledFusedMLP",
    "DynamicCompiledFusedMLP",
    "compile",
    "compile_fused_mlp",
    "dynamic_compile",
    "make_cache_key",
]
