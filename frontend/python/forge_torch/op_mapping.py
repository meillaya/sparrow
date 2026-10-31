"""Compatibility surface for the canonical op table.

The table moved to :mod:`forge_torch.ops`; this module keeps the Phase 1
names importable. New code should import from :mod:`forge_torch.ops`
directly.
"""

from __future__ import annotations

from forge_torch.ops import (
    DTYPE_TO_FORGE,
    SUPPORTED_ATEN_OPS,
    UNSUPPORTED_BUT_RECOGNISED,
    map_node_to_forge,
    normalize_target,
)

__all__ = (
    "DTYPE_TO_FORGE",
    "SUPPORTED_ATEN_OPS",
    "UNSUPPORTED_BUT_RECOGNISED",
    "map_node_to_forge",
    "normalize_target",
)
