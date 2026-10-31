#pragma once

// Sparrow Triton backend — Path A, CPU reference kernel (spec §7).
// This header is the *public* entry point used by `TritonProvider::launch`
// to materialize a fused GEMM + bias + GELU on the host. The shipped
// kernel is a naive triple-loop in fp32 with bias + GELU fused in the
// inner loop — correctness and readability over SIMD/threading. The
// public ABI of this header (struct shape + return type) is the stable
// contract between the kernel and the runtime.
#include <cstdint>

#include "forge_rt/result.hpp"

namespace forge_rt::backends::triton::cpu {

// All pointers are HOST-resident fp32 row-major buffers. The kernel does
// NOT validate dtype/layout at the pointer level — that is the provider's
// job (see TritonProvider::launch). The kernel still validates the
// *dimensional* contract: x is [M, K], w is [K, N], b is [N], y is [M, N].
struct FusedGemmBiasGeluInputs {
    const float* x;          // [M, K] row-major
    const float* w;          // [K, N] row-major
    const float* b;          // [N]
    float*       y;          // [M, N] row-major (output)
    int64_t M = 0;
    int64_t K = 0;
    int64_t N = 0;
};

// Naive triple-loop fp32 GEMM + bias + GELU (exact GELU matches
// torch.nn.functional.gelu(approximate="none")).
//
// Returns Result<void>:
//   * Error "non-positive dimension" if any of M/K/N <= 0
//   * Error "null pointer"       if x/w/b/y is null
//   * Error "K mismatch"         if the inner dim contract is violated
//                                 (kept as a sanity guard for callers that
//                                  build the struct from raw descriptors)
//
// Phase 1 supports fp32 only; the TritonProvider rejects other dtypes
// before reaching this kernel (the descriptor dtype check lives there).
Result<void> run_fused_gemm_bias_gelu(const FusedGemmBiasGeluInputs& in);

}  // namespace forge_rt::backends::triton::cpu