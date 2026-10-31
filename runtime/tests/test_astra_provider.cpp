// ForgeRT — AstraDeviceProvider gtest suite (spec §9, §10).
//
// The provider executes a binary artifact: microcode words plus a JSON
// header that binds plan inputs to DRAM offsets. This test hand-builds a
// K=8 16x16 tile with a non-zero bias and checks the result against a
// triple-loop reference.
#include "forge_rt/artifact_descriptor.hpp"
#include "forge_rt/buffer.hpp"
#include "forge_rt/providers/astra_provider.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
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

BufferSpec make_spec(const std::vector<int64_t>& shape, std::size_t byte_size) {
    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = shape;
    spec.descriptor.strides.assign(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1;) {
        spec.descriptor.strides[i - 1] = spec.descriptor.strides[i] * shape[i];
    }
    spec.descriptor.memory_space = MemorySpace::Host;
    spec.byte_size = byte_size;
    spec.alignment = 64;
    return spec;
}

std::uint64_t word(int opcode, int a, int b, int imm) {
    return (static_cast<std::uint64_t>(opcode) << 48) |
           (static_cast<std::uint64_t>(a) << 32) |
           (static_cast<std::uint64_t>(b) << 16) |
           static_cast<std::uint64_t>(imm);
}

std::vector<std::byte> bytes_of(const std::string& text) {
    std::vector<std::byte> out;
    out.reserve(text.size());
    for (const char c : text) out.push_back(std::byte{static_cast<unsigned char>(c)});
    return out;
}

}  // namespace

TEST(AstraProvider_BinaryExecutor, M16K8N16WithBiasAndGelu) {
    constexpr int64_t M = 16;
    constexpr int64_t K = 8;
    constexpr int64_t N = 16;
    constexpr int64_t kTileBytes = 16 * 16 * 4;

    // DRAM layout produced by the compiler for this shape.
    const std::size_t bias_offset = 0;
    const std::size_t a_offset = 64;
    const std::size_t b_offset = a_offset + static_cast<std::size_t>(M * K * 4);
    const std::size_t out_offset = 0;
    // SRAM layout.
    const std::size_t sram_acc = 0;
    const std::size_t sram_b = 2048;
    const std::size_t sram_bias = 3072;

    std::vector<std::uint64_t> words;
    words.push_back(word(0x01, static_cast<int>(sram_bias),
                         static_cast<int>(bias_offset), 64));
    for (int row = 0; row < 16; ++row) {
        words.push_back(word(0x01, static_cast<int>(sram_acc + row * K * 4),
                             static_cast<int>(a_offset + row * K * 4), K * 4));
    }
    words.push_back(word(0x01, static_cast<int>(sram_b),
                         static_cast<int>(b_offset), static_cast<int>(K * N * 4)));
    words.push_back(word(0x03, static_cast<int>(sram_acc),
                         static_cast<int>(sram_b), static_cast<int>(K)));
    words.push_back(word(0x06, static_cast<int>(sram_acc),
                         static_cast<int>(sram_bias), 256));
    words.push_back(word(0x04, static_cast<int>(sram_acc),
                         static_cast<int>(sram_acc), 256));
    words.push_back(word(0x02, static_cast<int>(out_offset),
                         static_cast<int>(sram_acc), static_cast<int>(kTileBytes)));
    words.push_back(word(0xFF, 0, 0, 0));

    nlohmann::json header = {
        {"version", 2},
        {"entry_point", "astra.microcode.v2"},
        {"dram_bytes", 4096},
        {"words", words},
        {"buffers",
         nlohmann::json::array(
             {{{"name", "x"}, {"role", "input"}, {"input_index", 0},
               {"dram_offset", a_offset}, {"byte_size", M * K * 4},
               {"shape", {M, K}}, {"dtype", "f32"}},
              {{"name", "w"}, {"role", "input"}, {"input_index", 1},
               {"dram_offset", b_offset}, {"byte_size", K * N * 4},
               {"shape", {K, N}}, {"dtype", "f32"}},
              {{"name", "b"}, {"role", "input"}, {"input_index", 2},
               {"dram_offset", bias_offset}, {"byte_size", N * 4},
               {"shape", {N}}, {"dtype", "f32"}},
              {{"name", "y"}, {"role", "output"},
               {"dram_offset", out_offset}, {"byte_size", kTileBytes},
               {"shape", {M, N}}, {"dtype", "f32"}}})},
        {"output", {{"name", "y"}, {"role", "output"},
                    {"dram_offset", out_offset}, {"byte_size", kTileBytes},
                    {"shape", {M, N}}, {"dtype", "f32"}}},
    };

    AstraProvider provider;
    auto x_spec = make_spec({M, K}, static_cast<std::size_t>(M * K * 4));
    auto w_spec = make_spec({K, N}, static_cast<std::size_t>(K * N * 4));
    auto b_spec = make_spec({N}, static_cast<std::size_t>(N * 4));
    auto y_spec = make_spec({M, N}, static_cast<std::size_t>(kTileBytes));
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

    std::mt19937 rng(0xCAFE'A5A5u);
    std::uniform_real_distribution<float> dist(-0.5F, 0.5F);
    std::vector<float> x(static_cast<std::size_t>(M * K));
    std::vector<float> w(static_cast<std::size_t>(K * N));
    std::vector<float> b(static_cast<std::size_t>(N));
    for (auto& value : x) value = dist(rng);
    for (auto& value : w) value = dist(rng);
    for (auto& value : b) value = dist(rng);
    std::memcpy(x_buf.data(), x.data(), x.size() * sizeof(float));
    std::memcpy(w_buf.data(), w.data(), w.size() * sizeof(float));
    std::memcpy(b_buf.data(), b.data(), b.size() * sizeof(float));

    ArtifactDescriptor artifact;
    artifact.target = TargetKind::Astra;
    artifact.entry_point = "astra.microcode.v2";
    artifact.binary = bytes_of(header.dump());

    std::vector<Buffer> inputs;
    inputs.push_back(std::move(x_buf));
    inputs.push_back(std::move(w_buf));
    inputs.push_back(std::move(b_buf));
    std::vector<Buffer> outputs;
    outputs.push_back(std::move(y_buf));

    auto launch = provider.launch(artifact, inputs, outputs, {});
    ASSERT_TRUE(launch.ok()) << launch.error().message;

    std::vector<float> got(static_cast<std::size_t>(M * N));
    std::memcpy(got.data(), outputs[0].data(), got.size() * sizeof(float));

    constexpr float kInvSqrt2 = 0.7071067811865475F;
    auto gelu = [](float value) {
        return 0.5F * value * (1.0F + std::erff(value * kInvSqrt2));
    };
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = b[static_cast<std::size_t>(n)];
            for (int64_t k = 0; k < K; ++k) {
                acc += x[static_cast<std::size_t>(m * K + k)] *
                       w[static_cast<std::size_t>(k * N + n)];
            }
            EXPECT_NEAR(got[static_cast<std::size_t>(m * N + n)], gelu(acc), 1e-4F);
        }
    }
}
