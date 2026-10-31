"""Sparrow Python frontend package."""

from __future__ import annotations

from forge_torch._version import __version__
from forge_torch.backend import (
    BACKEND,
    CompiledFusedMLP,
    CompiledGraph,
    compile,
    compile_fused_mlp,
    compile_graph,
)
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
    "CompiledGraph",
    "DynamicCompiledFusedMLP",
    "__version__",
    "compile",
    "compile_fused_mlp",
    "compile_graph",
    "dynamic_compile",
    "make_cache_key",
]
