// ForgeRT — TritonDeviceProvider implementation (spec §10, §7).
//
// Shipped scope: every operation runs on the host. allocate uses
// std::aligned_alloc, copy is std::memcpy, launch dispatches into the
// C++ CPU reference kernel at backends/triton/cpu/. The DeviceProvider
// ABI is the stable contract.
#include "forge_rt/providers/triton_provider.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "fused_gemm_bias_gelu.hpp"  // backends/triton/cpu/

namespace forge_rt::providers {

namespace {

// Round `n` up to the next multiple of `align`. `align` must be > 0.
// We require align to be a power-of-two for std::aligned_alloc.
std::size_t round_up_to(std::size_t n, std::size_t align) {
    // align is a power of two; use bit-twiddling.
    return (n + (align - 1)) & ~(align - 1);
}

// std::aligned_alloc requires alignment to be a supported power-of-two
// multiple of alignof(void*). If the caller asks for an alignment we
// can't honor (0 or not a power-of-two), we clamp to alignof(void*).
std::size_t clamp_alignment(uint64_t requested) {
    if (requested == 0) return alignof(void*);
    // Power-of-two check.
    if ((requested & (requested - 1)) != 0) {
        return alignof(void*);
    }
    return static_cast<std::size_t>(requested);
}

}  // namespace

TritonProvider::TritonProvider() = default;
TritonProvider::~TritonProvider() = default;

Result<Buffer> TritonProvider::allocate(const BufferSpec& spec) {
    if (spec.byte_size == 0) {
        return Error{"TritonProvider::allocate: zero byte_size", 2001};
    }

    const std::size_t align = clamp_alignment(spec.alignment);
    const std::size_t bytes = round_up_to(
        static_cast<std::size_t>(spec.byte_size), align);

    // std::aligned_alloc(size, alignment): size must be a multiple of
    // alignment. We've already rounded up; if size < alignment (rare but
    // legal for tiny buffers) we round up again to satisfy the contract.
    const std::size_t alloc_size = std::max(bytes, align);

    void* raw = std::aligned_alloc(align, alloc_size);
    if (raw == nullptr) {
        return Error{"TritonProvider::allocate: std::aligned_alloc(" +
                         std::to_string(alloc_size) + " bytes, align=" +
                         std::to_string(align) + ") returned nullptr",
                     2002};
    }

    // The caller asked for spec.byte_size; we allocated `alloc_size` to
    // satisfy std::aligned_alloc's alignment-multiple requirement. Update
    // the spec we hand to Buffer so downstream byte_size lookups reflect
    // the actual allocation.
    BufferSpec out_spec = spec;
    out_spec.byte_size = bytes;
    out_spec.alignment = align;

    return Buffer{raw, out_spec, /*owns=*/true};
}

Result<Event> TritonProvider::copy(Buffer& dst, const Buffer& src,
                                   std::span<const Event> /*deps*/) {
    if (src.spec().descriptor.memory_space != MemorySpace::Host ||
        dst.spec().descriptor.memory_space != MemorySpace::Host) {
        return Error{
            "TritonProvider::copy: only Host↔Host copies are supported "
            "(got src=" +
                std::string(to_string(src.spec().descriptor.memory_space)) +
                " dst=" +
                std::string(to_string(dst.spec().descriptor.memory_space)) +
                "); Device0/Device1 buffers are rejected by the shipped provider",
            2010};
    }

    if (src.data() == nullptr || dst.data() == nullptr) {
        return Error{"TritonProvider::copy: null buffer pointer", 2011};
    }

    const std::size_t nbytes = std::min(src.spec().byte_size, dst.spec().byte_size);
    if (nbytes == 0) {
        return Error{"TritonProvider::copy: zero-length copy", 2012};
    }

    std::memcpy(dst.data(), src.data(), nbytes);

    Event ev{next_event_id_++};
    ev.mark_completed();  // host copy is synchronous
    return ev;
}

Result<Event> TritonProvider::launch(const ArtifactDescriptor& artifact,
                                     std::span<const Buffer> inputs,
                                     std::span<Buffer> outputs,
                                     std::span<const Event> /*deps*/) {
    if (artifact.target != TargetKind::Triton) {
        return Error{
            "TritonProvider::launch: artifact.target must be TargetKind::Triton "
            "(got " +
                std::string(to_string(artifact.target)) + ")",
            2020};
    }

    // Contract: x, w, b in; y out.
    if (inputs.size() < 3) {
        return Error{"TritonProvider::launch: need 3 inputs (x, w, b); got " +
                         std::to_string(inputs.size()),
                     2021};
    }
    if (outputs.size() < 1) {
        return Error{"TritonProvider::launch: need 1 output (y); got " +
                         std::to_string(outputs.size()),
                     2022};
    }

    // Phase 1 dtype gate — only fp32. The kernel signature uses float*.
    auto check_fp32 = [](const Buffer& b, const char* name) -> Error {
        if (b.spec().descriptor.dtype != DataType::Float32) {
            return Error{std::string("TritonProvider::launch: ") + name +
                             " dtype must be Float32 (got " +
                             to_string(b.spec().descriptor.dtype) +
                             "); Phase 1 supports fp32 only",
                         2023};
        }
        return Error{};
    };
    if (auto e = check_fp32(inputs[0], "x"); e.code != 0) return e;
    if (auto e = check_fp32(inputs[1], "w"); e.code != 0) return e;
    if (auto e = check_fp32(inputs[2], "b"); e.code != 0) return e;
    if (auto e = check_fp32(outputs[0], "y"); e.code != 0) return e;

    // Shape contract: x=[M,K], w=[K,N], b=[N], y=[M,N].
    const auto& x_shape = inputs[0].spec().descriptor.shape;
    const auto& w_shape = inputs[1].spec().descriptor.shape;
    const auto& b_shape = inputs[2].spec().descriptor.shape;
    const auto& y_shape = outputs[0].spec().descriptor.shape;

    if (x_shape.size() != 2 || w_shape.size() != 2 ||
        b_shape.size() != 1 || y_shape.size() != 2) {
        return Error{
            "TritonProvider::launch: rank mismatch "
            "(x must be [M,K], w [K,N], b [N], y [M,N])",
            2024};
    }
    const int64_t M = x_shape[0];
    const int64_t K = x_shape[1];
    const int64_t N = w_shape[1];
    if (w_shape[0] != K) {
        return Error{"TritonProvider::launch: K mismatch x.shape[1]=" +
                         std::to_string(K) + " vs w.shape[0]=" +
                         std::to_string(w_shape[0]),
                     2025};
    }
    if (b_shape[0] != N) {
        return Error{"TritonProvider::launch: N mismatch w.shape[1]=" +
                         std::to_string(N) + " vs b.shape[0]=" +
                         std::to_string(b_shape[0]),
                     2026};
    }
    if (y_shape[0] != M || y_shape[1] != N) {
        return Error{"TritonProvider::launch: y shape [" +
                         std::to_string(y_shape[0]) + "," +
                         std::to_string(y_shape[1]) + "] != [M,N]=[" +
                         std::to_string(M) + "," + std::to_string(N) + "]",
                     2027};
    }

    // Build the kernel input struct and dispatch.
    backends::triton::cpu::FusedGemmBiasGeluInputs kern{};
    kern.x = static_cast<const float*>(inputs[0].data());
    kern.w = static_cast<const float*>(inputs[1].data());
    kern.b = static_cast<const float*>(inputs[2].data());
    kern.y = static_cast<float*>(outputs[0].data());
    kern.M = M;
    kern.K = K;
    kern.N = N;

    if (kern.x == nullptr || kern.w == nullptr ||
        kern.b == nullptr || kern.y == nullptr) {
        return Error{"TritonProvider::launch: null buffer data()", 2028};
    }

    auto kresult = backends::triton::cpu::run_fused_gemm_bias_gelu(kern);
    if (!kresult.ok()) {
        // Forward the kernel error with our launch prefix so the
        // scheduler log is self-describing.
        Error inner = kresult.error();
        return Error{"TritonProvider::launch: kernel: " + inner.message,
                     inner.code == 0 ? 2029 : inner.code};
    }

    Event ev{next_event_id_++};
    ev.mark_completed();  // CPU launch is synchronous
    return ev;
}

Result<void> TritonProvider::wait(const Event& /*event*/) {
    // Every event is marked completed at issue time, so wait() is
    // trivially satisfied.
    return Result<void>{};
}

}  // namespace forge_rt::providers