"""Compiler lowering registry.

``capabilities.py`` describes what a backend is built to run.
This module describes what the compiler can actually emit today. The
partitioner needs both: a backend that lists an op but has no lowering
must not be selected.

The shipped map is empty because the only real execution path is the
native fused-MLP kernel, which is a frontend pattern rather than a
general ``forge.tensor`` lowering. Empty sets make the partitioner fail
closed instead of planning a backend it cannot compile.
"""

from __future__ import annotations

from typing import Final

COMPILER_LOWERINGS: Final[dict[str, frozenset[str]]] = {
    "triton": frozenset(),
    "astra": frozenset(),
    "microgpu": frozenset(),
}

__all__ = ("COMPILER_LOWERINGS",)
