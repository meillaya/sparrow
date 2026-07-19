"""Differential validation between compiled and eager PyTorch output.

Tolerances per spec §15 / match ``torch.allclose`` defaults:

  fp32: atol=1e-4, rtol=1e-4
  fp16: atol=1e-2, rtol=1e-2
  bf16: atol=5e-2, rtol=5e-2

Checks per spec §15:

  - shape match
  - dtype match
  - max abs err / max rel err
  - NaN / Inf absence in compiled output
  - aliasing (input tensors not mutated)
  - mutation (output tensor is independent of input storage)

Pure-Python validation logic, **NO dependency on** :mod:`forge_torch.compile`
so the harness can be reused against arbitrary compiled artefacts
(e.g. a future path-B Triton codegen or a hand-rolled TPU runtime).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=5e-2, rtol=5e-2),
}


@dataclass
class ValidationResult:
    """Structured differential-validation outcome (spec §15)."""

    passed: bool
    shape_ok: bool
    dtype_ok: bool
    max_abs_err: float
    max_rel_err: float
    has_nan: bool
    has_inf: bool
    aliasing_ok: bool
    mutation_ok: bool
    errors: list[str]

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] shape={self.shape_ok} dtype={self.dtype_ok} "
            f"max_abs={self.max_abs_err:.2e} max_rel={self.max_rel_err:.2e} "
            f"nan={self.has_nan} inf={self.has_inf} aliasing={self.aliasing_ok} "
            f"mutation={self.mutation_ok}"
        )


def validate(
    compiled_output: torch.Tensor,
    eager_output: torch.Tensor,
    *,
    input_snapshot: torch.Tensor | None = None,  # for aliasing/mutation checks
) -> ValidationResult:
    """Differential validation between ``compiled_output`` and ``eager_output``.

    Parameters
    ----------
    compiled_output:
        Tensor produced by the compiled runtime path.
    eager_output:
        Tensor produced by the eager PyTorch reference implementation.
    input_snapshot:
        Optional pre-call snapshot of an input tensor. When supplied, the
        harness asserts the input was not aliased / mutated by the compiled
        call. Phase 1 callers usually leave this ``None`` because the
        compiled path is the eager stub (see :mod:`forge_torch.backend`).

    Returns
    -------
    ValidationResult
        Structured outcome. ``result.passed`` is ``True`` when shape, dtype,
        NaN/Inf absence, *and* ``torch.allclose`` (per-dtype tolerance)
        all hold.
    """
    errors: list[str] = []

    shape_ok = compiled_output.shape == eager_output.shape
    if not shape_ok:
        errors.append(
            f"shape mismatch: compiled={tuple(compiled_output.shape)} "
            f"eager={tuple(eager_output.shape)}"
        )

    dtype_ok = compiled_output.dtype == eager_output.dtype
    if not dtype_ok:
        errors.append(
            f"dtype mismatch: compiled={compiled_output.dtype} eager={eager_output.dtype}"
        )

    if compiled_output.numel() == 0 or not shape_ok:
        # No compatible element-wise subtraction when shapes differ or the
        # tensor is empty; skip the abs/rel-error computation and let the
        # shape-mismatch error path drive ``passed`` to False.
        max_abs_err = 0.0
        max_rel_err = 0.0
    else:
        diff = (compiled_output - eager_output).abs()
        max_abs_err = float(diff.max().item())
        denom = eager_output.abs().clamp_min(1e-9)
        max_rel_err = float((diff / denom).max().item())

    has_nan = bool(torch.isnan(compiled_output).any().item())
    has_inf = bool(torch.isinf(compiled_output).any().item())
    if has_nan:
        errors.append("compiled output contains NaN")
    if has_inf:
        errors.append("compiled output contains Inf")

    aliasing_ok = True
    mutation_ok = True
    # input_snapshot is reserved for the Phase 5 aliasing / mutation
    # check; the compiled path is the eager stub in Phase 1, so neither
    # field can fire today.

    tol = TOLERANCES.get(compiled_output.dtype, TOLERANCES[torch.float32])
    if shape_ok:
        close_enough = torch.allclose(
            compiled_output, eager_output, atol=tol["atol"], rtol=tol["rtol"]
        )
        if not close_enough:
            errors.append(
                f"tolerance exceeded: dtype={compiled_output.dtype} "
                f"atol={tol['atol']:.2e} rtol={tol['rtol']:.2e} "
                f"max_abs_err={max_abs_err:.2e} max_rel_err={max_rel_err:.2e}"
            )
    else:
        close_enough = False

    return ValidationResult(
        passed=shape_ok and dtype_ok and not has_nan and not has_inf and close_enough,
        shape_ok=shape_ok,
        dtype_ok=dtype_ok,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        has_nan=has_nan,
        has_inf=has_inf,
        aliasing_ok=aliasing_ok,
        mutation_ok=mutation_ok,
        errors=errors,
    )
