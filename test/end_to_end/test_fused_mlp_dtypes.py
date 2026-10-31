"""End-to-end FusedMLP dtype-differential pytest (Phase 2, Task 20).

Extends the Phase 1 fp32 differential coverage to fp16 + bf16. The
shipped :func:`forge_torch.backend.compile` delegates the runtime
dispatch to eager PyTorch, so the FX capture + MLIR import + cache key
generation must work for all three dtypes before the lowered kernel
(scheduled for Phase 5 — the C++ ``run_fused_gemm_bias_gelu`` kernel
is fp32-only today) takes over.

The reference for both ``compiled`` and ``numpy`` paths is
``torch.nn.functional.gelu(x @ w + b)``; the numpy reference computes
the same matmul + exact erf-GELU in fp64 then casts back to the test
dtype so the dtype-aware tolerance (spec §15: fp16 atol=1e-2, bf16
atol=5e-2) is meaningful.

The Phase 1 C++ CPU kernel (``backends/triton/cpu/fused_gemm_bias_gelu.cpp``)
is fp32-only today. fp16/bf16 support in the kernel is a Phase 5 TODO;
this test pins the *frontend pipeline* accepting those dtypes and the
*forward numerics* matching reference within dtype tolerance, which is
the contract the lowered kernel will have to meet when it lands.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from forge_torch.backend import compile
from forge_torch.conftest import FusedMLP
from forge_torch.validation import TOLERANCES

# Per spec §15 — fp16 atol=1e-2, bf16 atol=5e-2.
DTYPE_CASES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]


def _numpy_gelu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference GELU computed in fp64 then cast to the target dtype.

    Mirrors ``torch.nn.functional.gelu(approximate="none")`` exactly:
        y = 0.5 * (x @ w + b) * (1 + erf((x @ w + b) / sqrt(2)))
    """
    xf = x.detach().cpu().float().numpy().astype(np.float64)
    wf = w.detach().cpu().float().numpy().astype(np.float64)
    bf = b.detach().cpu().float().numpy().astype(np.float64)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    matmul = xf @ wf + bf
    erf_vec = np.vectorize(math.erf, otypes=[np.float64])
    gelu = 0.5 * matmul * (1.0 + erf_vec(matmul * inv_sqrt2))
    return torch.from_numpy(gelu.astype(np.float32)).to(x.dtype)


@pytest.mark.parametrize("torch_dtype", DTYPE_CASES)
def test_fused_mlp_dtype_compiled_matches_eager(torch_dtype: torch.dtype) -> None:
    """The full frontend pipeline accepts fp16/bf16 and matches eager within dtype atol."""
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(2, 16, 128, dtype=torch_dtype)
    w = torch.randn(128, 256, dtype=torch_dtype)
    b = torch.randn(256, dtype=torch_dtype)

    eager_out = model(x, w, b)
    compiled = compile(model, [x, w, b])
    compiled_out = compiled(x, w, b)

    tol = TOLERANCES[torch_dtype]
    assert compiled_out.dtype == torch_dtype
    assert compiled_out.shape == eager_out.shape
    assert torch.allclose(compiled_out, eager_out, atol=tol["atol"], rtol=tol["rtol"]), (
        f"compiled (eager-stub) output diverged from eager at {torch_dtype} "
        f"atol={tol['atol']} rtol={tol['rtol']}"
    )


@pytest.mark.parametrize("torch_dtype", DTYPE_CASES)
def test_fused_mlp_dtype_matches_numpy_reference(torch_dtype: torch.dtype) -> None:
    """The FusedMLP forward in fp16/bf16 matches a numpy GELU reference within dtype atol.

    Independent of ``torch.compile`` and the cached artifact — this test
    pins the dtype-agnostic numerical contract of the model's forward
    by comparing against a numpy reference computed via
    :func:`_numpy_gelu`. When the lowered fp16/bf16 kernel lands in
    Phase 5, this same contract will gate its correctness.

    Inputs are scaled to magnitude ±0.1 so the fp16/bf16 matmul output
    stays in a range where the dtype's ULP fits comfortably under the
    spec §15 atol — unmodified ``randn`` inputs push matmul values to
    ±50, where fp16's ULP is 2⁻⁷ ≈ 0.008 and bf16's is 2⁻⁵ ≈ 0.03,
    exceeding the tolerance band before any FusedMLP error accumulates.
    """
    torch.manual_seed(0)
    model = FusedMLP()
    x = (torch.randn(2, 16, 128, dtype=torch.float32) * 0.1).to(torch_dtype)
    w = (torch.randn(128, 256, dtype=torch.float32) * 0.1).to(torch_dtype)
    b = (torch.randn(256, dtype=torch.float32) * 0.1).to(torch_dtype)

    fmlp_out = model(x, w, b)
    ref_out = _numpy_gelu(x, w, b)

    tol = TOLERANCES[torch_dtype]
    assert fmlp_out.shape == ref_out.shape
    assert fmlp_out.dtype == torch_dtype
    diff = (fmlp_out.float() - ref_out.float()).abs()
    max_abs = float(diff.max().item())
    assert max_abs < tol["atol"], (
        f"FusedMLP fp16/bf16 output diverged from numpy GELU reference: "
        f"dtype={torch_dtype} max_abs_err={max_abs:.4e} atol={tol['atol']}"
    )


def test_fused_mlp_dtype_cache_key_routes_per_dtype(tmp_path) -> None:
    """fp32 / fp16 / bf16 of the same shape produce distinct cache keys (spec §13)."""
    from forge_torch.cache_store import CacheStore

    cache_root = tmp_path / "cache"
    store = CacheStore(cache_root)

    model = FusedMLP()
    keys: dict[torch.dtype, object] = {}
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, dtype=dt)
        w = torch.randn(64, 64, dtype=dt)
        b = torch.randn(64, dtype=dt)
        compiled = compile(model, [x, w, b], cache_store=store)
        keys[dt] = compiled.cache_key
        assert compiled.cache_key is not None
        assert store.get(compiled.cache_key) is not None, (
            f"compile() should have populated the cache for {dt}, but no artifact found"
        )
        assert compiled.cache_store is store

    digests = {dt: k.hexdigest() for dt, k in keys.items()}
    assert len(set(digests.values())) == 3, f"expected 3 distinct cache digests, got {digests}"
