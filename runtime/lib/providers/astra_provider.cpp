// ForgeRT — AstraDeviceProvider implementation (spec §9, §10).
//
// Launch executes a self-contained binary artifact:
//
//   ArtifactDescriptor.binary = UTF-8 JSON header
//     { version, entry_point, dram_bytes, words: [u64...],
//       buffers: [{name, role, dram_offset, byte_size, shape,
//                  input_index?}], output: {...} }
//
// The header binds each plan input to a DRAM offset. The provider copies
// the caller's buffers into a simulated DRAM image, executes the
// instruction words against a 512 KiB SRAM image, and copies the output
// region back. No child process or Python runtime is involved.
#include "forge_rt/providers/astra_provider.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "forge_rt/abi_types.hpp"

namespace forge_rt::providers {

namespace {

constexpr int kBadTarget = 2140;
constexpr int kBadArtifact = 2141;
constexpr int kBadArity = 2142;
constexpr int kBadDtype = 2143;
constexpr int kBadBuffer = 2144;
constexpr int kBadWord = 2145;
constexpr int kBadAddress = 2146;

constexpr std::size_t kSramBytes = 512 * 1024;
constexpr std::size_t kTileDim = 16;
constexpr std::size_t kTileElements = kTileDim * kTileDim;
constexpr std::size_t kTileBytes = kTileElements * sizeof(float);
constexpr int kDefaultDramBytes = 64 * 1024 * 1024;

constexpr int kOpDmaLoad = 0x01;
constexpr int kOpDmaStore = 0x02;
constexpr int kOpMatmul = 0x03;
constexpr int kOpVectorGelu = 0x04;
constexpr int kOpVectorAdd = 0x05;
constexpr int kOpVectorAddBroadcast = 0x06;
constexpr int kOpHalt = 0xFF;

std::size_t round_up_to(std::size_t n, std::size_t align) {
    return (n + (align - 1)) & ~(align - 1);
}

std::size_t clamp_alignment(uint64_t requested) {
    if (requested == 0) return alignof(void*);
    if ((requested & (requested - 1)) != 0) return alignof(void*);
    return static_cast<std::size_t>(requested);
}

float gelu_exact(float x) {
    return 0.5F * x * (1.0F + std::erf(x / std::sqrt(2.0F)));
}

bool ranges_fit(std::size_t offset, std::size_t bytes, std::size_t limit) {
    return offset <= limit && bytes <= limit - offset;
}

Result<void> execute_words(const std::vector<uint64_t>& words,
                           std::vector<std::byte>& dram,
                           std::vector<std::byte>& sram) {
    for (const uint64_t word : words) {
        const int opcode = static_cast<int>((word >> 48) & 0xFFFF);
        const std::size_t reg_a = static_cast<std::size_t>((word >> 32) & 0xFFFF);
        const std::size_t reg_b = static_cast<std::size_t>((word >> 16) & 0xFFFF);
        const std::size_t imm = static_cast<std::size_t>(word & 0x7FFF);
        switch (opcode) {
            case kOpDmaLoad: {
                if (!ranges_fit(reg_a, imm, sram.size())) {
                    return Error{"astra: DMA_LOAD SRAM range out of bounds", kBadAddress};
                }
                if (!ranges_fit(reg_b, imm, dram.size())) {
                    return Error{"astra: DMA_LOAD DRAM range out of bounds", kBadAddress};
                }
                std::memcpy(sram.data() + reg_a, dram.data() + reg_b, imm);
                break;
            }
            case kOpDmaStore: {
                if (!ranges_fit(reg_b, imm, sram.size())) {
                    return Error{"astra: DMA_STORE SRAM range out of bounds", kBadAddress};
                }
                if (!ranges_fit(reg_a, imm, dram.size())) {
                    return Error{"astra: DMA_STORE DRAM range out of bounds", kBadAddress};
                }
                std::memcpy(dram.data() + reg_a, sram.data() + reg_b, imm);
                break;
            }
            case kOpMatmul: {
                const std::size_t k = imm;
                if (k == 0 || k > kTileDim) {
                    return Error{"astra: MATMUL k must be in 1..16", kBadWord};
                }
                const std::size_t a_bytes = kTileDim * k * sizeof(float);
                const std::size_t b_bytes = k * kTileDim * sizeof(float);
                if (!ranges_fit(reg_a, std::max(a_bytes, kTileBytes), sram.size()) ||
                    !ranges_fit(reg_b, b_bytes, sram.size())) {
                    return Error{"astra: MATMUL operand range out of bounds", kBadAddress};
                }
                const auto* a = reinterpret_cast<const float*>(sram.data() + reg_a);
                const auto* b = reinterpret_cast<const float*>(sram.data() + reg_b);
                float product[kTileElements] = {};
                for (std::size_t m = 0; m < kTileDim; ++m) {
                    for (std::size_t n = 0; n < kTileDim; ++n) {
                        float sum = 0.0F;
                        for (std::size_t inner = 0; inner < k; ++inner) {
                            sum += a[m * k + inner] * b[inner * kTileDim + n];
                        }
                        product[m * kTileDim + n] = sum;
                    }
                }
                std::memcpy(sram.data() + reg_a, product, kTileBytes);
                break;
            }
            case kOpVectorGelu: {
                const std::size_t count = imm;
                if (!ranges_fit(reg_b, count * sizeof(float), sram.size()) ||
                    !ranges_fit(reg_a, count * sizeof(float), sram.size())) {
                    return Error{"astra: VECTOR_GELU range out of bounds", kBadAddress};
                }
                auto* dst = reinterpret_cast<float*>(sram.data() + reg_a);
                const auto* src = reinterpret_cast<const float*>(sram.data() + reg_b);
                for (std::size_t i = 0; i < count; ++i) dst[i] = gelu_exact(src[i]);
                break;
            }
            case kOpVectorAdd:
            case kOpVectorAddBroadcast: {
                const std::size_t count = imm;
                const std::size_t source_count =
                    opcode == kOpVectorAddBroadcast ? kTileDim : count;
                if (!ranges_fit(reg_b, source_count * sizeof(float), sram.size()) ||
                    !ranges_fit(reg_a, count * sizeof(float), sram.size())) {
                    return Error{"astra: VECTOR_ADD range out of bounds", kBadAddress};
                }
                auto* dst = reinterpret_cast<float*>(sram.data() + reg_a);
                const auto* src = reinterpret_cast<const float*>(sram.data() + reg_b);
                for (std::size_t i = 0; i < count; ++i) {
                    dst[i] += src[source_count == 0 ? 0 : i % source_count];
                }
                break;
            }
            case kOpHalt:
                return {};
            default:
                return Error{"astra: unknown opcode " + std::to_string(opcode), kBadWord};
        }
    }
    return {};
}

}  // namespace

AstraProvider::AstraProvider() = default;
AstraProvider::~AstraProvider() = default;

Result<Buffer> AstraProvider::allocate(const BufferSpec& spec) {
    if (spec.byte_size == 0) {
        return Error{"AstraProvider::allocate: zero byte_size", 2101};
    }
    const std::size_t align = clamp_alignment(spec.alignment);
    const std::size_t bytes = round_up_to(static_cast<std::size_t>(spec.byte_size), align);
    const std::size_t alloc_size = std::max(bytes, align);
    void* raw = std::aligned_alloc(align, alloc_size);
    if (raw == nullptr) {
        return Error{"AstraProvider::allocate: std::aligned_alloc failed", 2102};
    }
    BufferSpec out_spec = spec;
    out_spec.byte_size = bytes;
    out_spec.alignment = align;
    return Buffer{raw, out_spec, /*owns=*/true};
}

Result<Event> AstraProvider::copy(Buffer& dst, const Buffer& src,
                                  std::span<const Event> /*deps*/) {
    if (src.spec().descriptor.memory_space != MemorySpace::Host ||
        dst.spec().descriptor.memory_space != MemorySpace::Host) {
        return Error{"AstraProvider::copy: only Host<->Host copies are supported", 2110};
    }
    if (src.data() == nullptr || dst.data() == nullptr) {
        return Error{"AstraProvider::copy: null buffer pointer", 2111};
    }
    const std::size_t nbytes = std::min(src.spec().byte_size, dst.spec().byte_size);
    if (nbytes == 0) {
        return Error{"AstraProvider::copy: zero-length copy", 2112};
    }
    std::memcpy(dst.data(), src.data(), nbytes);
    Event ev{next_event_id_++};
    ev.mark_completed();
    return ev;
}

Result<Event> AstraProvider::launch(const ArtifactDescriptor& artifact,
                                    std::span<const Buffer> inputs,
                                    std::span<Buffer> outputs,
                                    std::span<const Event> /*deps*/) {
    if (artifact.target != TargetKind::Astra) {
        return Error{"AstraProvider::launch: artifact target must be Astra", kBadTarget};
    }
    if (artifact.entry_point != "astra.microcode.v2") {
        return Error{"AstraProvider::launch: expected entry point astra.microcode.v2",
                     kBadArtifact};
    }
    if (artifact.binary.empty()) {
        return Error{"AstraProvider::launch: empty artifact binary", kBadArtifact};
    }
    if (outputs.size() != 1) {
        return Error{"AstraProvider::launch: expected one output", kBadArity};
    }

    nlohmann::json header;
    try {
        const std::string text(reinterpret_cast<const char*>(artifact.binary.data()),
                               artifact.binary.size());
        header = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        return Error{"AstraProvider::launch: invalid header JSON: " +
                         std::string(error.what()),
                     kBadArtifact};
    }

    std::vector<uint64_t> words;
    std::size_t dram_bytes = kDefaultDramBytes;
    try {
        if (header.value("version", 0) != 2) {
            return Error{"AstraProvider::launch: unsupported artifact version", kBadArtifact};
        }
        words = header.at("words").get<std::vector<uint64_t>>();
        dram_bytes = header.value("dram_bytes", kDefaultDramBytes);
    } catch (const nlohmann::json::exception& error) {
        return Error{"AstraProvider::launch: bad header field: " +
                         std::string(error.what()),
                     kBadArtifact};
    }
    if (words.empty()) {
        return Error{"AstraProvider::launch: artifact has no instructions", kBadArtifact};
    }

    std::vector<std::byte> dram(dram_bytes, std::byte{0});
    std::vector<std::byte> sram(kSramBytes, std::byte{0});

    std::size_t output_offset = 0;
    std::size_t output_bytes = 0;
    try {
        for (const auto& buffer : header.at("buffers")) {
            const std::string role = buffer.at("role").get<std::string>();
            const std::size_t offset = buffer.at("dram_offset").get<std::size_t>();
            const std::size_t bytes = buffer.at("byte_size").get<std::size_t>();
            if (!ranges_fit(offset, bytes, dram.size())) {
                return Error{"AstraProvider::launch: buffer outside DRAM image", kBadBuffer};
            }
            if (role == "input") {
                const std::size_t index = buffer.at("input_index").get<std::size_t>();
                if (index >= inputs.size()) {
                    return Error{"AstraProvider::launch: missing plan input", kBadArity};
                }
                if (bytes > inputs[index].spec().byte_size) {
                    return Error{"AstraProvider::launch: input buffer too small", kBadBuffer};
                }
                std::memcpy(dram.data() + offset, inputs[index].data(), bytes);
            } else if (role == "output") {
                output_offset = offset;
                output_bytes = bytes;
            }
        }
        const auto& output = header.at("output");
        output_offset = output.at("dram_offset").get<std::size_t>();
        output_bytes = output.at("byte_size").get<std::size_t>();
    } catch (const nlohmann::json::exception& error) {
        return Error{"AstraProvider::launch: bad buffer binding: " +
                         std::string(error.what()),
                     kBadBuffer};
    }
    if (output_bytes == 0 || outputs[0].data() == nullptr) {
        return Error{"AstraProvider::launch: missing output binding", kBadBuffer};
    }
    if (output_bytes > outputs[0].spec().byte_size) {
        return Error{"AstraProvider::launch: output buffer too small", kBadBuffer};
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].spec().descriptor.dtype != DataType::Float32) {
            return Error{"AstraProvider::launch: inputs must be fp32", kBadDtype};
        }
    }

    auto executed = execute_words(words, dram, sram);
    if (!executed.ok()) return executed.error();

    std::memcpy(outputs[0].data(), dram.data() + output_offset, output_bytes);
    Event ev{next_event_id_++};
    ev.mark_completed();
    return ev;
}

Result<void> AstraProvider::wait(const Event& /*event*/) {
    return {};
}

}  // namespace forge_rt::providers
