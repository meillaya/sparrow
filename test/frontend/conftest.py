"""Test-local conftest — exposes the fixtures defined in
``forge_torch.conftest`` so pytest discovers them for files under
``test/frontend/``. Pytest only auto-loads ``conftest.py`` files from
ancestor directories of each test path; since
``frontend/python/forge_torch/conftest.py`` is not an ancestor of
``test/frontend/`` we re-export the fixtures here.
"""

from __future__ import annotations

from forge_torch.conftest import (
    FusedMLP,
    fixed_seed,
    fused_mlp,
    fused_mlp_inputs,
)

__all__ = ["FusedMLP", "fixed_seed", "fused_mlp", "fused_mlp_inputs"]
