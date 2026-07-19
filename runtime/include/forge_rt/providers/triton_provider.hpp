#pragma once

// ForgeRT — TritonDeviceProvider (spec §10, §7).
//
// Phase 1 "Triton" provider. It owns the host-side implementation of the
// DeviceProvider interface for the Triton target and delegates every
// launch into the C++ CPU reference kernel at
// `backends/triton/cpu/fused_gemm_bias_gelu`. Memory is host-resident
// (MemorySpace::Host); Phase 5 replaces `launch` with a real pybind11 →
// Triton kernel call and (optionally) Device0 allocations + cudaEvent_t
// fences. The DeviceProvider ABI surface is locked, so the swap is
// implementation-only.
//
// Spec §20 hard rules honored by this provider:
//   * Does NOT replace the CUDA allocator (we never allocate on Device0).
//   * Does NOT expose the Triton internal IR — Phase 5 will return a
//     compiled-kernel handle (see jit.py) and stop there.
//   * launch MAY call into the C++ CPU kernel for Phase 1.
#include <span>

#include "forge_rt/device_provider.hpp"

namespace forge_rt::providers {

class TritonProvider final : public DeviceProvider {
public:
    TritonProvider();
    ~TritonProvider() override;

    // Allocate a host-resident buffer honoring `spec.alignment`.
    // Returns Error on allocation failure (e.g. alignment > 0 but
    // std::aligned_alloc requires alignment to be a supported power-of-two
    // multiple of sizeof(void*)).
    Result<Buffer> allocate(const BufferSpec& spec) override;

    // Copy `src` into `dst`. Phase 1 only supports Host↔Host copies;
    // Device0/Device1 buffers return Error (Phase 5 widens this).
    Result<Event> copy(Buffer& dst, const Buffer& src,
                       std::span<const Event> deps) override;

    // Launch the compiled artifact. Phase 1 dispatches the
    // `fused_gemm_bias_gelu` entry point into the C++ CPU reference
    // kernel; Phase 5 replaces the body with a real pybind11 → Triton
    // kernel call. The returned Event is marked completed on return
    // (the CPU work is synchronous); Phase 5 will use a real fence.
    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override;

    // Wait is a no-op for Phase 1: every Event returned from
    // copy/launch is already marked completed. Phase 5 will block on
    // the underlying cudaEvent_t / HLS stream sync.
    Result<void> wait(const Event& event) override;

    MemorySpace memory_space() const override { return MemorySpace::Host; }
    TargetKind  target() const override { return TargetKind::Triton; }

private:
    // Monotonic event-id source. Phase 1 events are stubs; the id is
    // only used by Event::wait for diagnostic messages.
    uint64_t next_event_id_ = 1;
};

}  // namespace forge_rt::providers