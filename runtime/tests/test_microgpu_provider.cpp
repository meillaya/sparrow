// ForgeRT — MicroGpuDeviceProvider gtest suite (spec §11).
//
// Shipped scope: one functional correctness test for the
// `MicroGpuProvider::launch` path. The launch fires a Python child
// process that runs `backends.microgpu.interp.run` on the caller's
// input bytes + the artifact's 16-bit bytecode stream; this test
// asserts the resulting register file matches a hand-computed
// reference within elementwise tolerance.
//
// The microcode under test reads 4 input words from mem[0..3] into
// r0..r3, then writes them back reversed to mem[4..7]. The Python
// child emits the final register file as the output payload (one
// i32 per element); the test asserts r0..r3 hold the original
// input order.
#include "forge_rt/buffer.hpp"
#include "forge_rt/artifact_descriptor.hpp"
#include "forge_rt/providers/microgpu_provider.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using forge_rt::ArtifactDescriptor;
using forge_rt::Buffer;
using forge_rt::BufferSpec;
using forge_rt::DataType;
using forge_rt::MemorySpace;
using forge_rt::TargetKind;
using forge_rt::providers::MicroGpuProvider;

namespace {

// Build the 32-bit bytecode stream for the reverse-copy program. The
// wire layout is little-endian: four bytes per instruction word.
//
// Opcode layout (4-bit opcode, 6-bit dst, 6-bit src, 16-bit imm):
//   LOAD_GLOBAL  = 0x1   STORE_GLOBAL = 0x2   RETURN = 0x0
std::vector<std::byte> reverse_copy_program() {
    auto word_bytes = [](std::uint32_t word) {
        return std::vector<std::byte>{
            static_cast<std::byte>(word & 0xFF),
            static_cast<std::byte>((word >> 8) & 0xFF),
            static_cast<std::byte>((word >> 16) & 0xFF),
            static_cast<std::byte>((word >> 24) & 0xFF),
        };
    };
    auto load = [&](std::uint32_t dst, std::uint32_t addr) {
        return word_bytes((0x1u << 28) | (dst << 22) | addr);
    };
    auto store = [&](std::uint32_t src, std::uint32_t addr) {
        return word_bytes((0x2u << 28) | (src << 16) | addr);
    };
    auto ret = [&]() { return word_bytes(0x00000000u); };

    std::vector<std::byte> program;
    auto append = [&](const std::vector<std::byte>& chunk) {
        program.insert(program.end(), chunk.begin(), chunk.end());
    };
    append(load(0, 0x0));
    append(load(1, 0x1));
    append(load(2, 0x2));
    append(load(3, 0x3));
    append(store(3, 0x4));
    append(store(2, 0x5));
    append(store(1, 0x6));
    append(store(0, 0x7));
    append(ret());
    return program;
}

BufferSpec make_spec(const std::vector<int64_t>& shape,
                     std::size_t byte_size) {
    BufferSpec spec;
    spec.descriptor.dtype = DataType::Int32;
    spec.descriptor.shape = shape;
    spec.descriptor.strides = {1};
    spec.descriptor.memory_space = MemorySpace::Host;
    spec.descriptor.byte_offset = 0;
    spec.byte_size = byte_size;
    spec.alignment = 64;
    return spec;
}

std::string venv_python() {
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
    return "python3";
}

}  // namespace

TEST(MicroGpuProvider_LaunchFusedMLP, ReversesInputViaBytecode) {
    auto py = venv_python();
    MicroGpuProvider provider(py);

    constexpr int64_t kElements = 4;

    auto in_spec = make_spec({kElements * 2},
                             static_cast<std::size_t>(kElements * 2 * 4));
    auto out_spec = make_spec({kElements},
                              static_cast<std::size_t>(kElements * 4));

    auto in_res = provider.allocate(in_spec);
    auto out_res = provider.allocate(out_spec);
    ASSERT_TRUE(in_res.ok()) << in_res.error().message;
    ASSERT_TRUE(out_res.ok()) << out_res.error().message;

    Buffer in_buf = std::move(in_res.value());
    Buffer out_buf = std::move(out_res.value());

    std::vector<std::int32_t> input = {10, 20, 30, 40, 0, 0, 0, 0};
    std::memcpy(in_buf.data(), input.data(), input.size() * sizeof(std::int32_t));
    std::memset(out_buf.data(), 0, static_cast<std::size_t>(kElements * 4));

    ArtifactDescriptor ad;
    ad.target = TargetKind::MicroGPU;
    ad.entry_point = "microgpu_launch";
    ad.binary = reverse_copy_program();
    ad.arguments = {
        {"input", DataType::Int32, true, {kElements * 2}},
        {"output", DataType::Int32, false, {kElements}},
    };

    std::vector<Buffer> inputs;
    inputs.push_back(std::move(in_buf));
    std::vector<Buffer> outputs;
    outputs.push_back(std::move(out_buf));

    auto launch_res = provider.launch(ad, inputs, outputs, {});
    ASSERT_TRUE(launch_res.ok()) << launch_res.error().message;
    auto event_res = launch_res.value();
    auto wait_res = provider.wait(event_res);
    ASSERT_TRUE(wait_res.ok()) << wait_res.error().message;

    std::vector<std::int32_t> got(kElements);
    std::memcpy(got.data(), outputs[0].data(),
                got.size() * sizeof(std::int32_t));

    const std::vector<std::int32_t> expected = {10, 20, 30, 40};
    for (int64_t i = 0; i < kElements; ++i) {
        ASSERT_EQ(got[i], expected[i]) << "mismatch at i=" << i;
    }
}