// ForgeRT — CPU reference provider implementation.
//
// The kernels are deliberately simple, single-threaded fp32 loops. They
// exist to make the compiler's execution plan observable and correct
// before a machine-code backend lands.
#include "forge_rt/providers/cpu_provider.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace forge_rt::providers {

namespace {

constexpr int kBadTarget = 2301;
constexpr int kBadArtifact = 2302;
constexpr int kBadArity = 2303;
constexpr int kBadOutput = 2304;
constexpr int kBadDtype = 2305;
constexpr int kBadShape = 2306;
constexpr int kBadGemm = 2307;
constexpr int kBadBroadcast = 2308;
constexpr int kBadGelu = 2309;
constexpr int kBadSoftmax = 2310;
constexpr int kBadLayerNorm = 2311;
constexpr int kBadSdpa = 2312;
constexpr int kBadReshape = 2313;
constexpr int kUnknownEntryPoint = 2314;
constexpr int kBadCopy = 2315;

std::size_t round_up_to(std::size_t n, std::size_t align) {
    return (n + (align - 1)) & ~(align - 1);
}

std::size_t clamp_alignment(uint64_t requested) {
    if (requested == 0) return alignof(void*);
    if ((requested & (requested - 1)) != 0) return alignof(void*);
    return static_cast<std::size_t>(requested);
}

nlohmann::json parse_attributes(const ArtifactDescriptor& artifact) {
    if (artifact.binary.empty()) return nlohmann::json::object();
    const std::string text(reinterpret_cast<const char*>(artifact.binary.data()),
                           artifact.binary.size());
    return nlohmann::json::parse(text);
}

Result<void> check_float_buffer(const Buffer& buffer, const char* what) {
    if (buffer.data() == nullptr) {
        return Error{std::string(what) + ": null buffer", kBadDtype};
    }
    if (buffer.spec().descriptor.dtype != DataType::Float32) {
        return Error{std::string(what) + ": CPU kernels require fp32", kBadDtype};
    }
    if (!is_contiguous(buffer.spec().descriptor)) {
        return Error{std::string(what) + ": CPU kernels require contiguous buffers",
                     kBadShape};
    }
    if (buffer.spec().byte_size < byte_size(buffer.spec().descriptor)) {
        return Error{std::string(what) + ": buffer is smaller than its shape", kBadShape};
    }
    return {};
}

const float* as_float(const Buffer& buffer) {
    return static_cast<const float*>(buffer.data());
}

float* as_float(Buffer& buffer) {
    return static_cast<float*>(buffer.data());
}

std::size_t shape_numel(const std::vector<int64_t>& shape) {
    std::size_t total = 1;
    for (const int64_t dim : shape) total *= static_cast<std::size_t>(dim);
    return total;
}

bool same_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    return a == b;
}

float gelu_exact(float x) {
    return 0.5F * x * (1.0F + std::erf(x / std::sqrt(2.0F)));
}

float gelu_tanh(float x) {
    constexpr float kAlpha = 0.7978845608028654F;  // sqrt(2/pi)
    return 0.5F * x * (1.0F + std::tanh(kAlpha * (x + 0.044715F * x * x * x)));
}

float apply_activation(float x, const std::string& activation) {
    if (activation == "relu") return std::max(0.0F, x);
    if (activation == "gelu") return gelu_exact(x);
    if (activation == "silu") return x / (1.0F + std::exp(-x));
    return x;
}

Result<void> launch_gemm(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs) {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Error{"gemm: expected 2 inputs and 1 output", kBadArity};
    }
    for (const Buffer& buffer : inputs) {
        if (auto err = check_float_buffer(buffer, "gemm input"); !err) return err.error();
    }
    if (auto err = check_float_buffer(outputs[0], "gemm output"); !err) return err.error();

    const auto& a_shape = inputs[0].spec().descriptor.shape;
    const auto& b_shape = inputs[1].spec().descriptor.shape;
    const auto& out_shape = outputs[0].spec().descriptor.shape;
    if (b_shape.size() != 2 || a_shape.size() < 2) {
        return Error{"gemm: expected a rank >= 2 and b rank 2", kBadGemm};
    }

    const bool transpose_a = artifact.binary.empty()
                                 ? false
                                 : parse_attributes(artifact).value("transpose_a", false);
    const bool transpose_b = artifact.binary.empty()
                                 ? false
                                 : parse_attributes(artifact).value("transpose_b", false);

    const std::size_t a_rank = a_shape.size();
    const int64_t m = transpose_a ? a_shape[a_rank - 1] : a_shape[a_rank - 2];
    const int64_t k = transpose_a ? a_shape[a_rank - 2] : a_shape[a_rank - 1];
    const int64_t n = transpose_b ? b_shape[0] : b_shape[1];
    if ((transpose_b ? b_shape[1] : b_shape[0]) != k) {
        return Error{"gemm: inner dimensions do not match", kBadGemm};
    }

    std::vector<int64_t> expected(a_shape.begin(), a_shape.end() - 2);
    expected.push_back(m);
    expected.push_back(n);
    if (out_shape != expected) {
        return Error{"gemm: output shape does not match [..., M, N]", kBadGemm};
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return Error{"gemm: dimensions must be positive", kBadGemm};
    }

    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < a_rank; ++i) {
        batch *= static_cast<std::size_t>(a_shape[i]);
    }

    const float* a = as_float(inputs[0]);
    const float* b = as_float(inputs[1]);
    float* out = as_float(outputs[0]);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                float sum = 0.0F;
                for (int64_t inner = 0; inner < k; ++inner) {
                    const std::size_t a_index =
                        transpose_a
                            ? batch_index * static_cast<std::size_t>(k * m) +
                                  static_cast<std::size_t>(inner * m + row)
                            : batch_index * static_cast<std::size_t>(m * k) +
                                  static_cast<std::size_t>(row * k + inner);
                    const std::size_t b_index =
                        transpose_b
                            ? static_cast<std::size_t>(col * k + inner)
                            : static_cast<std::size_t>(inner * n + col);
                    sum += a[a_index] * b[b_index];
                }
                out[batch_index * static_cast<std::size_t>(m * n) +
                    static_cast<std::size_t>(row * n + col)] = sum;
            }
        }
    }
    return {};
}

std::size_t broadcast_offset(const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& output_coords) {
    std::size_t offset = 0;
    const std::size_t rank = shape.size();
    const std::size_t output_rank = output_coords.size();
    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t source_dim = shape[i];
        const std::size_t output_index = output_rank - rank + i;
        const std::size_t coordinate =
            source_dim == 1 ? 0 : static_cast<std::size_t>(output_coords[output_index]);
        offset = offset * static_cast<std::size_t>(source_dim) + coordinate;
    }
    return offset;
}

Result<void> launch_broadcast_add(std::span<const Buffer> inputs,
                                  std::span<Buffer> outputs) {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Error{"broadcast_add: expected 2 inputs and 1 output", kBadArity};
    }
    if (auto err = check_float_buffer(inputs[0], "broadcast_add lhs"); !err) return err.error();
    if (auto err = check_float_buffer(inputs[1], "broadcast_add rhs"); !err) return err.error();
    if (auto err = check_float_buffer(outputs[0], "broadcast_add output"); !err) return err.error();

    const auto& out_shape = outputs[0].spec().descriptor.shape;
    const auto& lhs_shape = inputs[0].spec().descriptor.shape;
    const auto& rhs_shape = inputs[1].spec().descriptor.shape;
    if (lhs_shape.size() > out_shape.size() || rhs_shape.size() > out_shape.size()) {
        return Error{"broadcast_add: operand rank exceeds output rank", kBadBroadcast};
    }
    const std::size_t total = shape_numel(out_shape);
    std::vector<int64_t> coords(out_shape.size(), 0);
    const float* lhs = as_float(inputs[0]);
    const float* rhs = as_float(inputs[1]);
    float* out = as_float(outputs[0]);
    for (std::size_t linear = 0; linear < total; ++linear) {
        std::size_t remainder = linear;
        for (std::size_t i = out_shape.size(); i-- > 0;) {
            coords[i] = static_cast<int64_t>(remainder % static_cast<std::size_t>(out_shape[i]));
            remainder /= static_cast<std::size_t>(out_shape[i]);
        }
        out[linear] = lhs[broadcast_offset(lhs_shape, coords)] +
                      rhs[broadcast_offset(rhs_shape, coords)];
    }
    return {};
}

Result<void> launch_gelu(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return Error{"gelu: expected 1 input and 1 output", kBadArity};
    }
    if (auto err = check_float_buffer(inputs[0], "gelu input"); !err) return err.error();
    if (auto err = check_float_buffer(outputs[0], "gelu output"); !err) return err.error();
    if (!same_shape(inputs[0].spec().descriptor.shape,
                    outputs[0].spec().descriptor.shape)) {
        return Error{"gelu: input and output shapes differ", kBadGelu};
    }
    const bool approximate =
        !artifact.binary.empty() && parse_attributes(artifact).value("approximate", false);
    const std::size_t total = shape_numel(inputs[0].spec().descriptor.shape);
    const float* input = as_float(inputs[0]);
    float* output = as_float(outputs[0]);
    for (std::size_t i = 0; i < total; ++i) {
        output[i] = approximate ? gelu_tanh(input[i]) : gelu_exact(input[i]);
    }
    return {};
}

Result<void> launch_fused_epilogue(const ArtifactDescriptor& artifact,
                                   std::span<const Buffer> inputs,
                                   std::span<Buffer> outputs) {
    if (inputs.size() < 1 || inputs.size() > 2 || outputs.size() != 1) {
        return Error{"fused_epilogue: expected 1 or 2 inputs and 1 output", kBadArity};
    }
    if (auto err = check_float_buffer(inputs[0], "fused_epilogue input"); !err) return err.error();
    if (inputs.size() == 2) {
        if (auto err = check_float_buffer(inputs[1], "fused_epilogue bias"); !err) return err.error();
    }
    if (auto err = check_float_buffer(outputs[0], "fused_epilogue output"); !err) return err.error();
    const auto& shape = inputs[0].spec().descriptor.shape;
    if (!same_shape(shape, outputs[0].spec().descriptor.shape)) {
        return Error{"fused_epilogue: output shape differs", kBadShape};
    }
    const std::string activation =
        artifact.binary.empty()
            ? "none"
            : parse_attributes(artifact).value("activation", std::string("none"));
    const std::size_t total = shape_numel(shape);
    const float* input = as_float(inputs[0]);
    const float* bias = inputs.size() == 2 ? as_float(inputs[1]) : nullptr;
    float* output = as_float(outputs[0]);
    const std::size_t last = shape.empty() ? 0 : static_cast<std::size_t>(shape.back());
    const std::size_t bias_elements =
        bias == nullptr ? 0 : shape_numel(inputs[1].spec().descriptor.shape);
    if (bias != nullptr && bias_elements != total && bias_elements != last) {
        return Error{"fused_epilogue: bias must be scalar, last-dim, or full shape",
                     kBadShape};
    }
    for (std::size_t i = 0; i < total; ++i) {
        float bias_value = 0.0F;
        if (bias != nullptr) {
            bias_value = bias_elements == total ? bias[i] : bias[i % last];
        }
        output[i] = apply_activation(input[i] + bias_value, activation);
    }
    return {};
}

Result<void> launch_softmax(const ArtifactDescriptor& artifact,
                            std::span<const Buffer> inputs,
                            std::span<Buffer> outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return Error{"softmax: expected 1 input and 1 output", kBadArity};
    }
    if (auto err = check_float_buffer(inputs[0], "softmax input"); !err) return err.error();
    if (auto err = check_float_buffer(outputs[0], "softmax output"); !err) return err.error();
    const auto& shape = inputs[0].spec().descriptor.shape;
    if (shape.empty() || !same_shape(shape, outputs[0].spec().descriptor.shape)) {
        return Error{"softmax: output shape differs", kBadShape};
    }
    const int dim = artifact.binary.empty()
                        ? -1
                        : parse_attributes(artifact).value("dim", -1);
    if (dim != -1 && dim != static_cast<int>(shape.size()) - 1) {
        return Error{"softmax: CPU kernels support the last dimension only", kBadSoftmax};
    }
    const std::size_t last = static_cast<std::size_t>(shape.back());
    if (last == 0) return {};
    const std::size_t rows = shape_numel(shape) / last;
    const float* input = as_float(inputs[0]);
    float* output = as_float(outputs[0]);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* in = input + row * last;
        float* out = output + row * last;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < last; ++i) maximum = std::max(maximum, in[i]);
        float sum = 0.0F;
        for (std::size_t i = 0; i < last; ++i) {
            out[i] = std::exp(in[i] - maximum);
            sum += out[i];
        }
        for (std::size_t i = 0; i < last; ++i) out[i] /= sum;
    }
    return {};
}

Result<void> launch_layer_norm(const ArtifactDescriptor& artifact,
                               std::span<const Buffer> inputs,
                               std::span<Buffer> outputs) {
    if (inputs.size() != 3 || outputs.size() != 1) {
        return Error{"layer_norm: expected 3 inputs and 1 output", kBadArity};
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (auto err = check_float_buffer(inputs[i], "layer_norm input"); !err) return err.error();
    }
    if (auto err = check_float_buffer(outputs[0], "layer_norm output"); !err) return err.error();
    const auto& shape = inputs[0].spec().descriptor.shape;
    if (shape.empty() || !same_shape(shape, outputs[0].spec().descriptor.shape)) {
        return Error{"layer_norm: output shape differs", kBadLayerNorm};
    }
    const std::size_t last = static_cast<std::size_t>(shape.back());
    if (last == 0) return Error{"layer_norm: last dimension is zero", kBadLayerNorm};
    if (shape_numel(inputs[1].spec().descriptor.shape) != last ||
        shape_numel(inputs[2].spec().descriptor.shape) != last) {
        return Error{"layer_norm: gamma and beta must have the last-dim length",
                     kBadLayerNorm};
    }
    const double epsilon =
        artifact.binary.empty()
            ? 1e-5
            : parse_attributes(artifact).value("epsilon", 1e-5);
    const std::size_t rows = shape_numel(shape) / last;
    const float* input = as_float(inputs[0]);
    const float* gamma = as_float(inputs[1]);
    const float* beta = as_float(inputs[2]);
    float* output = as_float(outputs[0]);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* in = input + row * last;
        float* out = output + row * last;
        double mean = 0.0;
        for (std::size_t i = 0; i < last; ++i) mean += in[i];
        mean /= static_cast<double>(last);
        double variance = 0.0;
        for (std::size_t i = 0; i < last; ++i) {
            const double delta = in[i] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(last);
        const double denominator = std::sqrt(variance + epsilon);
        for (std::size_t i = 0; i < last; ++i) {
            out[i] = static_cast<float>((in[i] - mean) / denominator) * gamma[i] + beta[i];
        }
    }
    return {};
}

Result<void> launch_sdpa(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs) {
    if (inputs.size() != 3 || outputs.size() != 1) {
        return Error{"sdpa: expected 3 inputs and 1 output", kBadArity};
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (auto err = check_float_buffer(inputs[i], "sdpa input"); !err) return err.error();
    }
    if (auto err = check_float_buffer(outputs[0], "sdpa output"); !err) return err.error();
    const auto& q_shape = inputs[0].spec().descriptor.shape;
    if (q_shape.size() != 3 || !same_shape(q_shape, inputs[1].spec().descriptor.shape) ||
        !same_shape(q_shape, inputs[2].spec().descriptor.shape) ||
        !same_shape(q_shape, outputs[0].spec().descriptor.shape)) {
        return Error{"sdpa: Q/K/V/output must be rank-3 and equally shaped", kBadSdpa};
    }
    const int heads = artifact.binary.empty()
                          ? 1
                          : parse_attributes(artifact).value("num_heads", 1);
    const std::size_t batch = static_cast<std::size_t>(q_shape[0]);
    const std::size_t seq = static_cast<std::size_t>(q_shape[1]);
    const std::size_t width = static_cast<std::size_t>(q_shape[2]);
    if (heads <= 0 || width % static_cast<std::size_t>(heads) != 0) {
        return Error{"sdpa: num_heads must divide the model width", kBadSdpa};
    }
    const std::size_t head_width = width / static_cast<std::size_t>(heads);
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_width));
    const float* q = as_float(inputs[0]);
    const float* k = as_float(inputs[1]);
    const float* v = as_float(inputs[2]);
    float* out = as_float(outputs[0]);
    std::vector<float> scores(seq);
    for (std::size_t b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (std::size_t i = 0; i < seq; ++i) {
                const float* query = q + (b * seq + i) * width + static_cast<std::size_t>(h) * head_width;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j < seq; ++j) {
                    const float* key = k + (b * seq + j) * width + static_cast<std::size_t>(h) * head_width;
                    float dot = 0.0F;
                    for (std::size_t d = 0; d < head_width; ++d) dot += query[d] * key[d];
                    scores[j] = dot * scale;
                    maximum = std::max(maximum, scores[j]);
                }
                float sum = 0.0F;
                for (std::size_t j = 0; j < seq; ++j) {
                    scores[j] = std::exp(scores[j] - maximum);
                    sum += scores[j];
                }
                float* output = out + (b * seq + i) * width + static_cast<std::size_t>(h) * head_width;
                for (std::size_t d = 0; d < head_width; ++d) output[d] = 0.0F;
                for (std::size_t j = 0; j < seq; ++j) {
                    const float weight = scores[j] / sum;
                    const float* value = v + (b * seq + j) * width + static_cast<std::size_t>(h) * head_width;
                    for (std::size_t d = 0; d < head_width; ++d) {
                        output[d] += weight * value[d];
                    }
                }
            }
        }
    }
    return {};
}

Result<void> launch_copy(std::span<const Buffer> inputs,
                         std::span<Buffer> outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return Error{"copy: expected 1 input and 1 output", kBadArity};
    }
    if (auto err = check_float_buffer(inputs[0], "copy input"); !err) return err.error();
    if (auto err = check_float_buffer(outputs[0], "copy output"); !err) return err.error();
    const std::size_t input_bytes = byte_size(inputs[0].spec().descriptor);
    const std::size_t output_bytes = byte_size(outputs[0].spec().descriptor);
    if (input_bytes != output_bytes) {
        return Error{"copy: input and output byte sizes differ", kBadCopy};
    }
    std::memcpy(outputs[0].data(), inputs[0].data(), input_bytes);
    return {};
}

Result<void> launch_reshape(std::span<const Buffer> inputs,
                            std::span<Buffer> outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return Error{"reshape: expected 1 input and 1 output", kBadArity};
    }
    if (auto err = check_float_buffer(inputs[0], "reshape input"); !err) return err.error();
    if (auto err = check_float_buffer(outputs[0], "reshape output"); !err) return err.error();
    if (shape_numel(inputs[0].spec().descriptor.shape) !=
        shape_numel(outputs[0].spec().descriptor.shape)) {
        return Error{"reshape: element counts differ", kBadReshape};
    }
    std::memcpy(outputs[0].data(), inputs[0].data(),
                byte_size(inputs[0].spec().descriptor));
    return {};
}

}  // namespace

CpuProvider::CpuProvider() = default;
CpuProvider::~CpuProvider() = default;

Result<Buffer> CpuProvider::allocate(const BufferSpec& spec) {
    if (spec.byte_size == 0) {
        return Error{"CpuProvider::allocate: zero byte_size", 2201};
    }
    const std::size_t align = clamp_alignment(spec.alignment);
    const std::size_t bytes = round_up_to(static_cast<std::size_t>(spec.byte_size), align);
    const std::size_t alloc_size = std::max(bytes, align);
    void* raw = std::aligned_alloc(align, alloc_size);
    if (raw == nullptr) {
        return Error{"CpuProvider::allocate: std::aligned_alloc returned nullptr", 2202};
    }
    BufferSpec out_spec = spec;
    out_spec.byte_size = bytes;
    out_spec.alignment = align;
    return Buffer{raw, out_spec, /*owns=*/true};
}

Result<Event> CpuProvider::copy(Buffer& dst, const Buffer& src,
                                std::span<const Event> /*deps*/) {
    if (src.spec().descriptor.memory_space != MemorySpace::Host ||
        dst.spec().descriptor.memory_space != MemorySpace::Host) {
        return Error{"CpuProvider::copy: only Host to Host copies are supported", 2203};
    }
    if (src.data() == nullptr || dst.data() == nullptr) {
        return Error{"CpuProvider::copy: null buffer pointer", 2204};
    }
    const std::size_t nbytes = std::min(src.spec().byte_size, dst.spec().byte_size);
    if (nbytes == 0) {
        return Error{"CpuProvider::copy: zero-length copy", 2205};
    }
    std::memcpy(dst.data(), src.data(), nbytes);
    Event event{next_event_id_++};
    event.mark_completed();
    return event;
}

Result<Event> CpuProvider::launch(const ArtifactDescriptor& artifact,
                                  std::span<const Buffer> inputs,
                                  std::span<Buffer> outputs,
                                  std::span<const Event> /*deps*/) {
    if (artifact.target != TargetKind::CPU) {
        return Error{"CpuProvider::launch: artifact target must be CPU", kBadTarget};
    }
    if (outputs.empty()) {
        return Error{"CpuProvider::launch: no output buffer", kBadOutput};
    }
    for (const Buffer& buffer : outputs) {
        if (buffer.spec().descriptor.dtype != DataType::Float32) {
            return Error{"CpuProvider::launch: CPU kernels require fp32 outputs",
                         kBadDtype};
        }
    }

    Result<void> result = [&]() -> Result<void> {
        try {
        if (artifact.entry_point == "forge.tensor.gemm") {
            return launch_gemm(artifact, inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.broadcast_add") {
            return launch_broadcast_add(inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.gelu") {
            return launch_gelu(artifact, inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.fused_epilogue") {
            return launch_fused_epilogue(artifact, inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.softmax") {
            return launch_softmax(artifact, inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.layer_norm") {
            return launch_layer_norm(artifact, inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.scaled_dot_product_attention") {
            return launch_sdpa(artifact, inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.reshape") {
            return launch_reshape(inputs, outputs);
        }
        if (artifact.entry_point == "forge.tensor.copy") {
            return launch_copy(inputs, outputs);
        }
        return Error{"CpuProvider::launch: unknown entry point " + artifact.entry_point,
                     kUnknownEntryPoint};
        } catch (const nlohmann::json::exception& error) {
            return Error{"CpuProvider::launch: invalid artifact JSON: " +
                             std::string(error.what()),
                         kBadArtifact};
        }
    }();

    if (!result) return result.error();
    Event event{next_event_id_++};
    event.mark_completed();
    return event;
}

Result<void> CpuProvider::wait(const Event& /*event*/) {
    return {};
}

}  // namespace forge_rt::providers
