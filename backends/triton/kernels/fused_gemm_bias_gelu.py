"""fused_gemm_bias_gelu — Triton kernel template (ForgeStack Path A).

Computes:  Y = gelu(X @ W + B)

Templated over:
  - BLOCK_M, BLOCK_N, BLOCK_K: tile sizes (default 64, 128, 32 per spec §7 example)
  - num_warps: warps per block (default 8)
  - num_stages: pipeline stages (default 4)
  - DTYPE: fp32 / fp16 / bf16 (cast target for the output)

The kernel takes X [B, S, K] or [M, K], W [K, N], B [N] and returns Y of the
same leading shape with last dim = N. Any-rank X is flattened to 2-D inside
the wrapper and reshaped back on the way out (the kernel itself is 2-D).

Spec reference: .omo/plans/forgestack.md §7 Path A.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gemm_bias_gelu_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] + k < K
        x = tl.load(
            x_ptrs,
            mask=k_mask & mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask & mask_n[:, None],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias and apply GELU (exact GELU; matches torch.nn.functional.gelu(approximate="none")).
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2))); 1/sqrt(2) ≈ 0.7071067811865475.
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(DTYPE), mask=mask_m[:, None] & mask_n[None, :])


def _resolve_triton_dtype(torch_dtype: torch.dtype) -> tl.dtype:
    """Map a torch dtype to the matching triton.language dtype."""
    if torch_dtype == torch.float32:
        return tl.float32
    if torch_dtype == torch.float16:
        return tl.float16
    if torch_dtype == torch.bfloat16:
        return tl.bfloat16
    raise TypeError(f"fused_gemm_bias_gelu: unsupported dtype {torch_dtype!r}")


def fused_gemm_bias_gelu(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 32,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Fused GEMM + bias + GELU. Returns a fresh tensor.

    x: [M, K] or [..., K]
    w: [K, N]
    b: [N]
    Returns: same shape as x with last dim = N.
    """
    assert x.is_cuda, (
        "Phase 1 Triton kernel requires CUDA; CPU fallback lives in backends/triton/cpu/"
    )
    assert w.shape[0] == x.shape[-1], (
        f"inner dim mismatch: x.shape[-1]={x.shape[-1]} vs w.shape[0]={w.shape[0]}"
    )
    assert b.shape[0] == w.shape[1], (
        f"bias shape mismatch: b.shape[0]={b.shape[0]} vs w.shape[1]={w.shape[1]}"
    )
    assert x.dtype == w.dtype == b.dtype, (
        "fused_gemm_bias_gelu requires x, w, b to share the same dtype"
    )

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = w.shape[1]
    y_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_gemm_bias_gelu_kernel[grid](
        x_2d,
        w,
        b,
        y_2d,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        w.stride(0),
        w.stride(1),
        y_2d.stride(0),
        y_2d.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        DTYPE=_resolve_triton_dtype(x.dtype),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y_2d.reshape(*orig_shape[:-1], N)
