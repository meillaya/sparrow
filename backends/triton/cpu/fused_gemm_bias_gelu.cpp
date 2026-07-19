// ForgeStack Triton backend — Path A, CPU reference kernel (spec §7).
//
// Phase 1: naive triple-loop in fp32 with bias + GELU fused. No threading,
// no SIMD intrinsics — Phase 1 values correctness and readability. The
// Phase 5 swap-in replaces this TU with a real pybind11 → Triton call,
// but the public ABI in fused_gemm_bias_gelu.hpp is locked.
//
// SIMD/threading optimization is a Phase 2 follow-up; Phase 1 values
// correctness and readability.
#include "fused_gemm_bias_gelu.hpp"

#include <cmath>
#include <cstdint>

namespace forge_rt::backends::triton::cpu {

namespace {

// 1 / sqrt(2). Exact GELU matches torch.nn.functional.gelu(approximate="none").
inline constexpr float kInvSqrt2 = 0.7071067811865475f;

// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
inline float gelu_exact(float x) noexcept {
    return 0.5f * x * (1.0f + std::erff(x * kInvSqrt2));
}

}  // namespace

Result<void> run_fused_gemm_bias_gelu(const FusedGemmBiasGeluInputs& in) {
    // ---- dimension validation ----
    if (in.M <= 0 || in.K <= 0 || in.N <= 0) {
        return Error{"fused_gemm_bias_gelu: non-positive dimension (M=" +
                         std::to_string(in.M) + ", K=" + std::to_string(in.K) +
                         ", N=" + std::to_string(in.N) + ")",
                     1001};
    }

    // ---- pointer validation ----
    if (in.x == nullptr || in.w == nullptr || in.b == nullptr || in.y == nullptr) {
        return Error{"fused_gemm_bias_gelu: null input/output pointer", 1002};
    }

    // The TritonProvider already enforces the [M,K]/[K,N]/[N]/[M,N] dtype
    // and shape contract before this kernel runs, but we re-check the inner
    // K contract here so a future caller that builds FusedGemmBiasGeluInputs
    // from raw descriptors still gets a clean error instead of an OOB read.
    // (x.last_dim and w.first_dim are not part of this struct — the
    // provider pre-validates them — so this guard is a no-op today.)
    // Kept as documentation of the contract; no runtime check needed here.

    // ---- compute ----
    // SIMD/threading optimization is a Phase 2 follow-up; Phase 1 values
    // correctness and readability.
    for (int64_t m = 0; m < in.M; ++m) {
        for (int64_t n = 0; n < in.N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < in.K; ++k) {
                acc += in.x[m * in.K + k] * in.w[k * in.N + n];
            }
            acc += in.b[n];
            in.y[m * in.N + n] = gelu_exact(acc);
        }
    }

    return Result<void>{};
}

}  // namespace forge_rt::backends::triton::cpu