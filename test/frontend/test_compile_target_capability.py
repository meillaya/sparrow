"""Phase 7 ``compile(target=...)`` capability-check pytest (Task 5, plan §Todos).

Pins the spec §6 wiring between :func:`forge_torch.backend.compile` and
:data:`forge_torch.capabilities.BACKEND_CAPABILITIES`:

* Default target is ``"triton"`` — existing Phase 1-8 callers (no
  ``target=`` argument) must continue to work unchanged.
* ``target="astra"`` on a model containing ``aten.softmax`` must emit a
  :class:`UserWarning` naming ``forge.tensor.softmax`` (Astra's microcode
  ISA lacks it).
* ``target="triton"`` on a model that uses only gemm / broadcast_add /
  gelu must emit no warning (Triton supports every Phase 7 op).

Coverage (Task 5 acceptance = 3 tests; we ship exactly those 3).
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from forge_torch.backend import compile


def test_default_target_is_triton(fused_mlp_inputs) -> None:
    """``compile(model, inputs)`` keeps Phase 1-8 behaviour: no warning, eager result.

    The default ``target="triton"`` path is exercised by every shipped
    Phase 1-8 caller (including ``test_fx_importer.py``); it must remain
    a no-op relative to the pre-Phase-7 contract.
    """
    from forge_torch.conftest import FusedMLP

    x, w, b = fused_mlp_inputs

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        compiled = compile(FusedMLP(), [x, w, b])

    got = compiled(x, w, b)
    eager = FusedMLP()(x, w, b)
    assert torch.allclose(got, eager, atol=1e-5)
    assert got.shape == eager.shape
    assert got.dtype == eager.dtype


def test_target_astra_emits_warning_for_unsupported_op() -> None:
    """``target="astra"`` + softmax model emits a UserWarning naming the op.

    Astra's microcode ISA does not include softmax
    (``backends/astra/sim/timing.py`` has no ``SIGMOID`` / ``SOFTMAX``
    instruction), so the capability check must surface this mismatch as
    a :class:`UserWarning` rather than silently dropping the op.

    Phase 7 Tasks 12 + 14 wired ``aten.softmax`` into ``forge.tensor.softmax``
    + the FX importer, so the compiled path now succeeds (delegates to
    eager per Phase 1 backwards-compat) — we verify the capability
    warning still fires *before* the eager fallback runs.
    """

    class HasSoftmax(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.softmax(x, dim=-1)

    m = HasSoftmax()
    x = torch.randn(2, 16, dtype=torch.float32)

    with pytest.warns(UserWarning, match="forge.tensor.softmax"):
        compiled = compile(m, [x], target="astra")
        out = compiled(x)
        eager = m(x)
        assert torch.allclose(out, eager, atol=1e-5)


def test_target_triton_no_warning_for_supported_ops() -> None:
    """``target="triton"`` + gemm/broadcast_add/gelu-only model emits no warning.

    Triton supports every Phase 7 forge.tensor op, so the capability
    check must produce no warnings on a model that exercises only the
    canonical gemm + bias + activation triple. The strict
    ``simplefilter("error", ...)`` promotes any emitted :class:`UserWarning`
    to an exception so the test fails the moment a regression sneaks one
    in.
    """

    class GemmAddGelu(nn.Module):
        def forward(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return F.gelu(x @ w + b)

    m = GemmAddGelu()
    x = torch.randn(2, 8, 16, dtype=torch.float32)
    w = torch.randn(16, 32, dtype=torch.float32)
    b = torch.randn(32, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        compiled = compile(m, [x, w, b], target="triton")

    out = compiled(x, w, b)
    assert out.shape == (2, 8, 32)
    assert out.dtype == torch.float32
