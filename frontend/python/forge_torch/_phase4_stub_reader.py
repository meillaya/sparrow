"""Reader for the Phase 4 CompilationReport.alternative_astra_cost / alternative_fpga_cost stubs (Phase 7 Wave 2)."""

# frontend/python/forge_torch/_phase4_stub_reader.py
#
# Phase 4 cost-stub reader (Task 9 of `forgestack-phase7.md`).
#
# Phase 4's `CompilationReport` carries the two ``alternative_*_cost``
# strings (e.g. ``"1.31x"``, ``"2.04x including transfer"``) that record
# how much slower the alternative backends would have been vs. the selected
# target. The Phase 7 cost model (Task 7) consumes those strings as
# downstream scaling factors on the analytical cycle estimate.
#
# This module is the canonical reader: it returns the stub dict keyed by
# *target name* (``"astra"`` → ``"triton_int4"``) rather than by the
# dataclass field names used by `forge_torch._phase7_cycle_tables`, and
# filters out empty strings so a freshly-instantiated report yields an
# empty dict. The cost model treats an empty dict as "no scaling signal
# available" rather than silently multiplying by an empty factor.

from __future__ import annotations

from forge_torch.report import CompilationReport


def read_phase4_stubs(report: CompilationReport) -> dict[str, str]:
    """Return the Phase 4 ``alternative_*_cost`` strings as a target-keyed dict.

    The keys are the canonical backend target names the Phase 7 cost
    model consumes:

    * ``"astra"`` — maps to ``CompilationReport.alternative_astra_cost``
      (e.g. the ratio when the model could have been lowered onto Astra).
    * ``"triton_int4"`` — maps to ``CompilationReport.alternative_fpga_cost``
      (the historical int4 FPGA ratio).

    Empty values are filtered out so a default `CompilationReport`
    yields an empty dict (``{}``) and the partitioner can treat the
    report as "no historical scaling signal" without ever computing
    against an empty factor.
    """
    candidates: dict[str, str] = {
        "astra": report.alternative_astra_cost,
        "triton_int4": report.alternative_fpga_cost,
    }
    return {target: ratio for target, ratio in candidates.items() if ratio}


__all__ = ("read_phase4_stubs",)
