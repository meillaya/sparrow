"""Phase 4 cost-stub reader pytest (Task 9, plan §Todos).

Pins the read-side contract that the Phase 7 CostModel (Task 7) consumes
from a `CompilationReport` — the ``alternative_astra_cost`` /
``alternative_fpga_cost`` strings the partitioner multiplies onto the
analytical cycle estimate to fold in measured historical ratios.

Coverage (Task 9 acceptance = 2 tests; we ship exactly those 2):

  * Known `CompilationReport` returns the stub dict keyed by *target name*
    (``"astra"`` / ``"triton_int4"``).
  * A freshly-instantiated report yields an empty dict (no scaling signal).
"""

from __future__ import annotations

from frontend.python.forge_torch._phase4_stub_reader import read_phase4_stubs
from frontend.python.forge_torch.report import CompilationReport


def test_read_phase4_stubs_returns_targeted_dict_for_known_report() -> None:
    """`read_phase4_stubs` returns ``{"astra": ..., "triton_int4": ...}``.

    Both keys come from the report's ``alternative_*_cost`` attributes.
    The keys are *target names* (not the dataclass field names used by
    `forge_torch._phase7_cycle_tables.read_phase4_stubs`) so the cost
    model can index the stub dict by backend name directly.
    """
    report = CompilationReport(
        model_name="FusedMLP",
        alternative_astra_cost="1.31x",
        alternative_fpga_cost="2.04x including transfer",
    )

    stubs = read_phase4_stubs(report)

    assert stubs == {
        "astra": "1.31x",
        "triton_int4": "2.04x including transfer",
    }


def test_read_phase4_stubs_returns_empty_dict_for_default_report() -> None:
    """A default `CompilationReport` (no alternative costs set) → empty dict.

    Empty cost strings are filtered out so the cost model's scaling step
    never silently multiplies by an empty factor; an empty stub dict is
    the contract for "no historical scaling signal available".
    """
    report = CompilationReport(model_name="<unset>")

    stubs = read_phase4_stubs(report)

    assert stubs == {}
