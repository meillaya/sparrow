"""Pytest fixtures for end-to-end tests (Task 13).

Re-exports the ``FusedMLP`` module + ``fused_mlp_inputs`` fixture from
:mod:`forge_torch.conftest` so tests under ``test/end_to_end/`` can
build the canonical Phase-1 model without each test file importing
the package-level conftest directly.
"""

from __future__ import annotations

from forge_torch.conftest import (
    FusedMLP,
    fixed_seed,
    fused_mlp,
    fused_mlp_inputs,
)

__all__ = ["FusedMLP", "fixed_seed", "fused_mlp", "fused_mlp_inputs"]
