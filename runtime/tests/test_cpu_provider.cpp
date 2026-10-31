// Tests for the CPU reference provider and its kernel registry.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge_rt/providers/cpu_provider.hpp"

namespace forge_rt::providers {
namespace {

BufferSpec make_spec(std::vector<int64_t> shape) {
    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = std::move(shape);
    spec.descriptor.strides.assign(spec.descriptor.shape.size(), 1);
    for (std::size_t i = spec.descriptor.shape.size(); i-- > 1;) {
        spec.descriptor.strides[i - 1] =
            spec.descriptor.strides[i] * spec.descriptor.shape[i];
    }
    spec.byte_size = forge_rt::byte_size(spec.descriptor);
    spec.alignment = 64;
    return spec;
}

Buffer view(std::vector<float>& data, std::vector<int64_t> shape) {
    return Buffer{data.data(), make_spec(std::move(shape)), /*owns=*/false};
}

ArtifactDescriptor artifact(std::string entry_point, std::string attributes = "{}") {
    ArtifactDescriptor result;
    result.target = TargetKind::CPU;
    result.entry_point = std::move(entry_point);
    result.binary.reserve(attributes.size());
    for (const char value : attributes) {
        result.binary.push_back(std::byte{static_cast<unsigned char>(value)});
    }
    return result;
}

TEST(CpuProvider, GemmMatchesReference) {
    std::vector<float> a{1, 2, 3, 4, 5, 6};  // 2x3
    std::vector<float> b{7, 8, 9, 10, 11, 12};  // 3x2
    std::vector<float> out(4, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(a, {2, 3}));
    inputs.push_back(view(b, {3, 2}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {2, 2}));

    CpuProvider provider;
    const auto result = provider.launch(artifact("forge.tensor.gemm"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_FLOAT_EQ(out[0], 58.0F);
    EXPECT_FLOAT_EQ(out[1], 64.0F);
    EXPECT_FLOAT_EQ(out[2], 139.0F);
    EXPECT_FLOAT_EQ(out[3], 154.0F);
}

TEST(CpuProvider, GemmHandlesBatchedInput) {
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};  // 2x2x2
    std::vector<float> b{1, 0, 0, 1};              // 2x2 identity
    std::vector<float> out(8, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(a, {2, 2, 2}));
    inputs.push_back(view(b, {2, 2}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {2, 2, 2}));

    CpuProvider provider;
    const auto result = provider.launch(artifact("forge.tensor.gemm"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_EQ(out, a);
}

TEST(CpuProvider, BroadcastAddSupportsTrailingBias) {
    std::vector<float> x{1, 2, 3, 4};
    std::vector<float> bias{10, 20};
    std::vector<float> out(4, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {2, 2}));
    inputs.push_back(view(bias, {2}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {2, 2}));

    CpuProvider provider;
    const auto result =
        provider.launch(artifact("forge.tensor.broadcast_add"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_FLOAT_EQ(out[0], 11.0F);
    EXPECT_FLOAT_EQ(out[1], 22.0F);
    EXPECT_FLOAT_EQ(out[2], 13.0F);
    EXPECT_FLOAT_EQ(out[3], 24.0F);
}

TEST(CpuProvider, GeluUsesExactErfByDefault) {
    std::vector<float> x{-1.0F, 0.0F, 1.0F};
    std::vector<float> out(3, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {3}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {3}));

    CpuProvider provider;
    const auto result = provider.launch(artifact("forge.tensor.gelu"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_NEAR(out[0], -0.158655, 1e-5);
    EXPECT_FLOAT_EQ(out[1], 0.0F);
    EXPECT_NEAR(out[2], 0.841345, 1e-5);
}

TEST(CpuProvider, SoftmaxNormalizesEachRow) {
    std::vector<float> x{1, 2, 3, 1, 2, 3};
    std::vector<float> out(6, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {2, 3}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {2, 3}));

    CpuProvider provider;
    const auto result = provider.launch(artifact("forge.tensor.softmax"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_NEAR(out[0] + out[1] + out[2], 1.0F, 1e-6);
    EXPECT_NEAR(out[0], out[3], 1e-6);
}

TEST(CpuProvider, LayerNormNormalizesLastDim) {
    std::vector<float> x{1, 2, 3, 4};
    std::vector<float> gamma{1, 1};
    std::vector<float> beta{0, 0};
    std::vector<float> out(4, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {2, 2}));
    inputs.push_back(view(gamma, {2}));
    inputs.push_back(view(beta, {2}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {2, 2}));

    CpuProvider provider;
    const auto result =
        provider.launch(artifact("forge.tensor.layer_norm", R"({"epsilon":0.0})"), inputs,
                        outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_NEAR(out[0], -1.0F, 1e-5);
    EXPECT_NEAR(out[1], 1.0F, 1e-5);
    EXPECT_NEAR(out[2], -1.0F, 1e-5);
    EXPECT_NEAR(out[3], 1.0F, 1e-5);
}

TEST(CpuProvider, FusedEpilogueAddsBiasAndActivation) {
    std::vector<float> x{-1, 0, 1};
    std::vector<float> bias{0, 0, 0};
    std::vector<float> out(3, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {3}));
    inputs.push_back(view(bias, {3}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {3}));

    CpuProvider provider;
    const auto result = provider.launch(
        artifact("forge.tensor.fused_epilogue", R"({"activation":"relu"})"), inputs,
        outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_FLOAT_EQ(out[0], 0.0F);
    EXPECT_FLOAT_EQ(out[1], 0.0F);
    EXPECT_FLOAT_EQ(out[2], 1.0F);
}

TEST(CpuProvider, GemmHandlesBatchedSecondOperand) {
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};  // 2x2x2
    std::vector<float> b{1, 0, 0, 1, 2, 0, 0, 2};  // 2x2x2 (identity, 2*identity)
    std::vector<float> out(8, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(a, {2, 2, 2}));
    inputs.push_back(view(b, {2, 2, 2}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {2, 2, 2}));

    CpuProvider provider;
    const auto result = provider.launch(artifact("forge.tensor.gemm"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_EQ(out, (std::vector<float>{1, 2, 3, 4, 10, 12, 14, 16}));
}

TEST(CpuProvider, MulBroadcastsTrailingOperand) {
    std::vector<float> x{1, 2, 3, 4};
    std::vector<float> scale{10, 20};
    std::vector<float> out(4, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {2, 2}));
    inputs.push_back(view(scale, {2}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {2, 2}));

    CpuProvider provider;
    const auto result = provider.launch(artifact("forge.tensor.mul"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_EQ(out, (std::vector<float>{10, 40, 30, 80}));
}

TEST(CpuProvider, ReluClampsNegatives) {
    std::vector<float> x{-2.0F, -0.5F, 0.0F, 3.0F};
    std::vector<float> out(4, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {4}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {4}));

    CpuProvider provider;
    const auto result = provider.launch(artifact("forge.tensor.relu"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_EQ(out, (std::vector<float>{0.0F, 0.0F, 0.0F, 3.0F}));
}

TEST(CpuProvider, TransposeAppliesPermutation) {
    std::vector<float> x{1, 2, 3, 4, 5, 6};  // 2x3
    std::vector<float> out(6, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {2, 3}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {3, 2}));

    CpuProvider provider;
    const auto result = provider.launch(
        artifact("forge.tensor.transpose", R"({"permutation":[1,0]})"), inputs,
        outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_EQ(out, (std::vector<float>{1, 4, 2, 5, 3, 6}));
}

TEST(CpuProvider, SelectDropsIndexedDimension) {
    std::vector<float> x{1, 2, 3, 4, 5, 6};  // 2x3
    std::vector<float> out(3, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {2, 3}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {3}));

    CpuProvider provider;
    const auto result = provider.launch(
        artifact("forge.tensor.select", R"({"dim":0,"index":1})"), inputs, outputs, {});
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_EQ(out, (std::vector<float>{4, 5, 6}));
}

TEST(CpuProvider, RejectsWrongTarget) {
    std::vector<float> x{1};
    std::vector<float> out(1, 0.0F);
    std::vector<Buffer> inputs;
    inputs.push_back(view(x, {1}));
    std::vector<Buffer> outputs;
    outputs.push_back(view(out, {1}));
    ArtifactDescriptor wrong = artifact("forge.tensor.gelu");
    wrong.target = TargetKind::Triton;

    CpuProvider provider;
    const auto result = provider.launch(wrong, inputs, outputs, {});
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, 2301);
}

}  // namespace
}  // namespace forge_rt::providers
