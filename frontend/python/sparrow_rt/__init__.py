"""Compatibility shim for the old native module name ``_sparrow_rt``."""

from __future__ import annotations

import importlib

try:
    _native = importlib.import_module("_sparrow_rt")
except ImportError as exc:  # pragma: no cover - exercised without the build
    raise ImportError(
        "sparrow_rt requires the native _sparrow_rt extension; build it with "
        "`cmake -B build -S . -G Ninja && cmake --build build` and set "
        "PYTHONPATH=build/python"
    ) from exc

execute_plan = _native.execute_plan
run_fused_mlp = _native.run_fused_mlp

__all__ = ("execute_plan", "run_fused_mlp")
