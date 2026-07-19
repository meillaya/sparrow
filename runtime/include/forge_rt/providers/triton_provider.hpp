#pragma once

// ForgeRT — TritonDeviceProvider (spec §10, §7).
//
// Shipped scope: the "Triton" provider owns the host-side implementation
// of the DeviceProvider interface for the Triton target and delegates
// every launch into the C++ CPU reference kernel at
// `backends/triton/cpu/fused_gemm_bias_gelu`. Memory is host-resident
// (MemorySpace::Host). The DeviceProvider ABI surface is locked.
//
// Spec §20 hard rules honored by this provider:
//   * Does NOT replace the CUDA allocator (we never allocate on Device0).
//   * Does NOT expose the Triton internal IR — only a compiled-kernel
//     handle (see jit.py) is returned.
//   * launch calls into the C++ CPU reference kernel.
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

    // Copy `src` into `dst`. Only Host↔Host copies are supported;
    // Device0/Device1 buffers return Error.
    Result<Event> copy(Buffer& dst, const Buffer& src,
                       std::span<const Event> deps) override;

    // Launch the compiled artifact. Dispatches the
    // `fused_gemm_bias_gelu` entry point into the C++ CPU reference
    // kernel. The returned Event is marked completed on return because
    // the CPU work is synchronous.
    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override;

    // Wait is a no-op today: every Event returned from copy/launch is
    // already marked completed.
    Result<void> wait(const Event& event) override;

    MemorySpace memory_space() const override { return MemorySpace::Host; }
    TargetKind  target() const override { return TargetKind::Triton; }

private:
    // Monotonic event-id source. Events are stubs; the id is only used
    // by Event::wait for diagnostic messages.
    uint64_t next_event_id_ = 1;
};

}  // namespace forge_rt::providers