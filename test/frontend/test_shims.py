"""Compatibility shims for the pre-0.3 package names."""

from __future__ import annotations

import importlib

import pytest


def test_sparrow_torch_reexports_forge_torch() -> None:
    import sparrow_torch

    import forge_torch

    assert sparrow_torch.__version__ == forge_torch.__version__
    assert sparrow_torch.compile is forge_torch.compile
    backend = importlib.import_module("sparrow_torch.backend")
    assert backend.compile is forge_torch.compile


def test_sparrow_rt_reexports_the_native_module() -> None:
    pytest.importorskip("_sparrow_rt")
    import importlib

    import sparrow_rt

    _sparrow_rt = importlib.import_module("_sparrow_rt")

    assert sparrow_rt.execute_plan is _sparrow_rt.execute_plan
    assert sparrow_rt.run_fused_mlp is _sparrow_rt.run_fused_mlp
