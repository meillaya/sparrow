// ForgeRT — AstraDeviceProvider gtest suite (spec §10, §9).
//
// Shipped scope: one functional correctness test for the
// `AstraProvider::launch` path. The launch fires a Python child
// process that runs `backends.astra.sim.run_functional` on the
// caller's input bytes; this test asserts the resulting tensor
// matches a hand-computed reference within 1e-4 elementwise (gelu
// via libm's `erff` agrees with the simulator's fp32 `erf` to
// fp32-rounding noise).
//
// Tile shape: M=N=16 (the only shape the shipped TILE-only matmul
// can serve). Bias is zero, since the simulator's ISA has no vector
// add — a non-zero bias is guarded against at runtime with exit
// code 2 and surfaces as a launch error in the C++ provider.
#include "forge_rt/buffer.hpp"
#include "forge_rt/artifact_descriptor.hpp"
#include "forge_rt/providers/astra_provider.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using forge_rt::ArtifactDescriptor;
using forge_rt::Buffer;
using forge_rt::BufferSpec;
using forge_rt::DataType;
using forge_rt::MemorySpace;
using forge_rt::TargetKind;
using forge_rt::providers::AstraProvider;

namespace {

// Build a host-resident fp32 BufferSpec with the given shape/strides
// (caller passes them explicitly since we have several distinct shapes
// in one test binary).
BufferSpec make_tile_spec(const std::vector<int64_t>& shape,
                          const std::vector<int64_t>& strides,
                          std::size_t byte_size) {
    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = shape;
    spec.descriptor.strides = strides;
    spec.descriptor.memory_space = MemorySpace::Host;
    spec.descriptor.byte_offset = 0;
    spec.byte_size = byte_size;
    spec.alignment = 64;
    return spec;
}

// Resolve `.venv/bin/python` next to the runtime build's project root.
// The astra child process needs the project on its import path so the
// `backends.*` and `compiler.lib.Dialect.Astra.*` imports resolve.
std::string venv_python() {
    // Walk upward from cwd looking for the `.venv/bin/python` that the
    // scripts/setup-env.sh helpers install. We can't rely on stripping
    // `/runtime/...` because ctest's cwd is `build/runtime/tests`, two
    // segments past the project root — the substring lies inside the
    // build dir, not at the project boundary.
    std::array<char, 4096> cwd{};
    if (::getcwd(cwd.data(), cwd.size()) == nullptr) return "python3";

    std::string path(cwd.data());
    for (int i = 0; i < 8; ++i) {
        std::string candidate = path + "/.venv/bin/python";
        if (::access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
        auto slash = path.find_last_of('/');
        if (slash == std::string::npos || slash == 0) break;
        path.erase(slash);
    }
    // Fallback: hope the test runner set PATH to include `.venv/bin`.
    return "python3";
}

}  // namespace

// ---------------------------------------------------------------------------
// launch_fused_mlp_matches_naive_reference — provider matches a hand-rolled
// triple-loop within 1e-4 elementwise. Shape: M=16 K=8 N=16, bias=0.
// ---------------------------------------------------------------------------
TEST(AstraProvider_LaunchFusedMLP, M16K8N16BiasZero) {
    // Pin the interpreter to the project's venv so subprocess.run can
    // import backends.astra.sim + compiler.lib.Dialect.Astra.*.
    auto py = venv_python();
    AstraProvider provider(py);

    constexpr int64_t M = 16;
    constexpr int64_t K = 8;
    constexpr int64_t N = 16;

    // Deterministic RNG seed so the test is reproducible.
    std::mt19937 rng(0xCAFE'A5A5u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Inputs: x[M,K], w[K,N], b[N]=zeros. The shipped astra path
    // has no vector-add op, so the test exercises the bias=0 contract.
    auto x_spec = make_tile_spec({M, K}, {K, 1},
                                 static_cast<std::size_t>(M * K * sizeof(float)));
    auto w_spec = make_tile_spec({K, N}, {N, 1},
                                 static_cast<std::size_t>(K * N * sizeof(float)));
    auto b_spec = make_tile_spec({N}, {1},
                                 static_cast<std::size_t>(N * sizeof(float)));
    auto y_spec = make_tile_spec({M, N}, {N, 1},
                                 static_cast<std::size_t>(M * N * sizeof(float)));

    auto x_res = provider.allocate(x_spec);
    auto w_res = provider.allocate(w_spec);
    auto b_res = provider.allocate(b_spec);
    auto y_res = provider.allocate(y_spec);
    ASSERT_TRUE(x_res.ok()) << x_res.error().message;
    ASSERT_TRUE(w_res.ok()) << w_res.error().message;
    ASSERT_TRUE(b_res.ok()) << b_res.error().message;
    ASSERT_TRUE(y_res.ok()) << y_res.error().message;

    Buffer x_buf = std::move(x_res.value());
    Buffer w_buf = std::move(w_res.value());
    Buffer b_buf = std::move(b_res.value());
    Buffer y_buf = std::move(y_res.value());

    std::vector<float> x(M * K);
    std::vector<float> w(K * N);
    std::vector<float> b(N, 0.0f);  // bias=0 by contract
    for (auto& v : x) v = dist(rng);
    for (auto& v : w) v = dist(rng);

    std::memcpy(x_buf.data(), x.data(), x.size() * sizeof(float));
    std::memcpy(w_buf.data(), w.data(), w.size() * sizeof(float));
    std::memcpy(b_buf.data(), b.data(), b.size() * sizeof(float));
    std::memset(y_buf.data(), 0, static_cast<std::size_t>(M * N * sizeof(float)));

    ArtifactDescriptor ad;
    ad.target = TargetKind::Astra;
    ad.entry_point = "astra.reference.tile16";
    // 8-byte opaque marker keeps the artifact non-empty; the provider
    // does not yet read microcode bytes back from the artifact (it
    // builds the program on the Python side).
    ad.binary.assign(1, std::byte{0x00});
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
    ASSERT_TRUE(wait_res.ok()) << wait_res.error().message;

    // Read provider output.
    std::vector<float> got(M * N);
    std::memcpy(got.data(), outputs[0].data(), got.size() * sizeof(float));

    // Hand-computed reference: triple-loop matmul, add zero bias, gelu.
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
            ref[m * N + n] = gelu(acc);  // bias = 0
        }
    }

    // Elementwise compare with a generous tolerance — the simulator
    // uses fp32 erff which agrees with libm's erff to fp32-rounding
    // noise on a 16x16 tile.
    constexpr float kTol = 1e-4f;
    for (int64_t i = 0; i < M * N; ++i) {
        ASSERT_NEAR(got[i], ref[i], kTol) << "mismatch at i=" << i;
    }
}
