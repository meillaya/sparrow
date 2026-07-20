"""Public Astra simulator entry point."""

# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, assert_never

import numpy as np
from numpy.typing import NDArray

from backends.astra.sim.functional import last_dram_tensor, run_functional
from backends.astra.sim.timing import SimMetrics, run_timing

Mode = Literal["functional", "timing"]

__all__: Final = (
    "Mode",
    "SimMetrics",
    "run_microcode",
)


def run_microcode(
    words: Sequence[int],
    *,
    mode: Mode = "functional",
    metrics: bool = False,
) -> tuple[NDArray[np.float32] | None, SimMetrics | None]:
    """Run packed microcode and optionally return timing metrics."""
    match mode:
        case "functional":
            return last_dram_tensor(run_functional(words)), None
        case "timing":
            result, timing_metrics = run_timing(words)
            return last_dram_tensor(result), timing_metrics if metrics else None
        case unreachable:
            assert_never(unreachable)
