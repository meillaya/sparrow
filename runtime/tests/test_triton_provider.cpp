// ForgeRT — TritonDeviceProvider gtest suite (spec §10, §7).
//
// Three tests covering the three Task 11 acceptance criteria:
//   1. allocate_release — buffer round-trip + safe destruction.
//   2. launch_gemm_matches_naive_reference — elementwise correctness
//      against an independently-computed reference. This is the key
//      correctness test for the kernel.
//   3. launch_incompatible_dtype_errors — fp16 inputs are rejected with
//      a "fp32 only" error message (Phase 1 dtype gate).
#include "forge_rt/buffer.hpp"
#include "forge_rt/artifact_descriptor.hpp"
#include "forge_rt/providers/triton_provider.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using forge_rt::ArtifactDescriptor;
using forge_rt::Buffer;
using forge_rt::BufferSpec;
using forge_rt::DataType;
using forge_rt::MemorySpace;
using forge_rt::TargetKind;
using forge_rt::providers::TritonProvider;

// Build a host-resident fp32 BufferSpec with the given byte size and
// alignment. Defaults to alignof(void*) (the floor std::aligned_alloc
// requires).
static BufferSpec make_host_f32_spec(std::size_t bytes, std::size_t align = 0) {
    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = {static_cast<int64_t>(bytes / sizeof(float))};
    spec.descriptor.strides = {1};
    spec.descriptor.memory_space = MemorySpace::Host;
    spec.descriptor.byte_offset = 0;
    spec.byte_size = bytes;
    spec.alignment = align;
    return spec;
}

// ---------------------------------------------------------------------------
// 1. allocate_release — 1 KiB buffer, write known pattern, read back, free.
// ---------------------------------------------------------------------------
TEST(TritonProvider_allocate_release, WriteReadAndSafeDestruct) {
    TritonProvider provider;

    constexpr std::size_t kBytes = 1024;
    auto spec = make_host_f32_spec(kBytes, /*align=*/64);

    auto buf_result = provider.allocate(spec);
    ASSERT_TRUE(buf_result.ok()) << buf_result.error().message;
    Buffer buf = std::move(buf_result.value());

    ASSERT_NE(buf.data(), nullptr);
    ASSERT_TRUE(buf.owns());

    // Write a known byte pattern via memset — every byte is 0xAB.
    std::memset(buf.data(), 0xAB, kBytes);

    // Read back via byte access; first/last/middle should all be 0xAB.
    auto* bytes = static_cast<unsigned char*>(buf.data());
    EXPECT_EQ(bytes[0], 0xAB);
    EXPECT_EQ(bytes[kBytes / 2], 0xAB);
    EXPECT_EQ(bytes[kBytes - 1], 0xAB);

    // Confirm the buffer was aligned to the requested 64-byte boundary.
    // (The TritonProvider clamps non-power-of-two to alignof(void*); 64
    // is a power of two so the request is honored.)
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buf.data()) % 64u, 0u);

    // buf goes out of scope here; ~Buffer calls std::free on the
    // aligned_alloc'd region. There is no crash or double-free. If a
    // sanitizer build is active (CI uses -fsanitize=address) a leak or
    // double-free would surface here; this test asserts structural
    // ownership semantics.
}

// ---------------------------------------------------------------------------
// 2. launch_gemm_matches_naive_reference — provider matches a hand-rolled
// triple-loop within 1e-5 elementwise. Shapes: M=4 K=6 N=8.
// ---------------------------------------------------------------------------
TEST(TritonProvider_launch_gemm_matches_naive_reference, M4K6N8) {
    TritonProvider provider;

    constexpr int64_t M = 4;
    constexpr int64_t K = 6;
    constexpr int64_t N = 8;

    // Deterministic RNG seed so the test is reproducible.
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Allocate input/output buffers on the host.
    auto x_spec = make_host_f32_spec(static_cast<std::size_t>(M * K) * sizeof(float));
    x_spec.descriptor.shape = {M, K};
    x_spec.descriptor.strides = {K, 1};
    auto w_spec = make_host_f32_spec(static_cast<std::size_t>(K * N) * sizeof(float));
    w_spec.descriptor.shape = {K, N};
    w_spec.descriptor.strides = {N, 1};
    auto b_spec = make_host_f32_spec(static_cast<std::size_t>(N) * sizeof(float));
    b_spec.descriptor.shape = {N};
    b_spec.descriptor.strides = {1};
    auto y_spec = make_host_f32_spec(static_cast<std::size_t>(M * N) * sizeof(float));
    y_spec.descriptor.shape = {M, N};
    y_spec.descriptor.strides = {N, 1};

    auto x_res = provider.allocate(x_spec);
    auto w_res = provider.allocate(w_spec);
    auto b_res = provider.allocate(b_spec);
    auto y_res = provider.allocate(y_spec);
    ASSERT_TRUE(x_res.ok());
    ASSERT_TRUE(w_res.ok());
    ASSERT_TRUE(b_res.ok());
    ASSERT_TRUE(y_res.ok());

    Buffer x_buf = std::move(x_res.value());
    Buffer w_buf = std::move(w_res.value());
    Buffer b_buf = std::move(b_res.value());
    Buffer y_buf = std::move(y_res.value());

    // Fill inputs with random fp32 values; reference reuses these same
    // vectors so the test stays self-contained.
    std::vector<float> x(M * K);
    std::vector<float> w(K * N);
    std::vector<float> b(N);
    for (auto& v : x) v = dist(rng);
    for (auto& v : w) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    std::memcpy(x_buf.data(), x.data(), x.size() * sizeof(float));
    std::memcpy(w_buf.data(), w.data(), w.size() * sizeof(float));
    std::memcpy(b_buf.data(), b.data(), b.size() * sizeof(float));
    // Zero the output before launch to make sure the kernel actually wrote.
    std::memset(y_buf.data(), 0, static_cast<std::size_t>(M * N) * sizeof(float));

    // Build the artifact descriptor — target Triton, entry_point can be
    // any string in Phase 1 (the provider doesn't dispatch on it yet).
    ArtifactDescriptor ad;
    ad.target = TargetKind::Triton;
    ad.entry_point = "fused_gemm_bias_gelu";
    ad.binary.resize(16);  // opaque kernel artifact (Phase 1 stub)
    ad.arguments = {
        {"x", DataType::Float32, true,  {M, K}},
        {"w", DataType::Float32, true,  {K, N}},
        {"b", DataType::Float32, true,  {N}},
        {"y", DataType::Float32, false, {M, N}},
    };

    std::vector<Buffer> inputs;
    inputs.push_back(std::move(x_buf));
    inputs.push_back(std::move(w_buf));
    inputs.push_back(std::move(b_buf));
    std::vector<Buffer> outputs;
    outputs.push_back(std::move(y_buf));

    auto launch_res = provider.launch(ad, inputs, outputs, {});
    ASSERT_TRUE(launch_res.ok()) << launch_res.error().message;
    auto event_res = launch_res.value();
    auto wait_res = provider.wait(event_res);
    ASSERT_TRUE(wait_res.ok());

    // Read provider output.
    std::vector<float> got(M * N);
    std::memcpy(got.data(), outputs[0].data(), got.size() * sizeof(float));

    // Compute independent reference.
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    auto gelu = [](float v) {
        return 0.5f * v * (1.0f + std::erff(v * kInvSqrt2));
    };
    std::vector<float> ref(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += x[m * K + k] * w[k * N + n];
            }
            acc += b[n];
            ref[m * N + n] = gelu(acc);
        }
    }

    // Elementwise compare with a generous tolerance — the kernel and the
    // reference both use the same erff path, so the difference is fp32
    // rounding noise (well below 1e-5).
    constexpr float kTol = 1e-5f;
    for (int64_t i = 0; i < M * N; ++i) {
        ASSERT_NEAR(got[i], ref[i], kTol) << "mismatch at i=" << i;
    }
}

// ---------------------------------------------------------------------------
// 3. launch_incompatible_dtype_errors — fp16 inputs are rejected with a
// message that mentions "fp32 only".
// ---------------------------------------------------------------------------
TEST(TritonProvider_launch_incompatible_dtype_errors, Float16InputsRejected) {
    TritonProvider provider;

    constexpr int64_t M = 2;
    constexpr int64_t K = 3;
    constexpr int64_t N = 4;

    auto x_spec = make_host_f32_spec(static_cast<std::size_t>(M * K) * sizeof(float));
    x_spec.descriptor.shape = {M, K};
    x_spec.descriptor.strides = {K, 1};
    x_spec.descriptor.dtype = DataType::Float16;  // ← wrong dtype

    auto x_res = provider.allocate(x_spec);
    ASSERT_TRUE(x_res.ok());

    // Allocate the other buffers at the *correct* dtype (Float32); only x
    // is fp16 so the dtype check fires first and the test stays tight.
    auto w_spec = make_host_f32_spec(static_cast<std::size_t>(K * N) * sizeof(float));
    w_spec.descriptor.shape = {K, N};
    w_spec.descriptor.strides = {N, 1};
    auto b_spec = make_host_f32_spec(static_cast<std::size_t>(N) * sizeof(float));
    b_spec.descriptor.shape = {N};
    auto y_spec = make_host_f32_spec(static_cast<std::size_t>(M * N) * sizeof(float));
    y_spec.descriptor.shape = {M, N};
    y_spec.descriptor.strides = {N, 1};

    auto w_res = provider.allocate(w_spec);
    auto b_res = provider.allocate(b_spec);
    auto y_res = provider.allocate(y_spec);
    ASSERT_TRUE(w_res.ok());
    ASSERT_TRUE(b_res.ok());
    ASSERT_TRUE(y_res.ok());

    Buffer x_buf = std::move(x_res.value());
    Buffer w_buf = std::move(w_res.value());
    Buffer b_buf = std::move(b_res.value());
    Buffer y_buf = std::move(y_res.value());

    ArtifactDescriptor ad;
    ad.target = TargetKind::Triton;
    ad.entry_point = "fused_gemm_bias_gelu";
    ad.binary.resize(16);
    ad.arguments = {
        {"x", DataType::Float32, true,  {M, K}},
        {"w", DataType::Float32, true,  {K, N}},
        {"b", DataType::Float32, true,  {N}},
        {"y", DataType::Float32, false, {M, N}},
    };

    std::vector<Buffer> inputs;
    inputs.push_back(std::move(x_buf));
    inputs.push_back(std::move(w_buf));
    inputs.push_back(std::move(b_buf));
    std::vector<Buffer> outputs;
    outputs.push_back(std::move(y_buf));

    auto launch_res = provider.launch(ad, inputs, outputs, {});
    ASSERT_FALSE(launch_res.ok());
    EXPECT_NE(launch_res.error().message.find("fp32 only"), std::string::npos)
        << "error message did not mention the fp32-only policy: '"
        << launch_res.error().message << "'";
}