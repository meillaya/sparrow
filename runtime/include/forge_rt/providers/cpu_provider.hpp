#pragma once

// ForgeRT — CPU reference provider.
//
// Executes plan artifacts whose entry point names a kernel in the
// Sparrow CPU registry. Memory is host-resident. The provider is
// synchronous: launch computes the result and returns a completed Event.
#include <atomic>
#include <cstdint>
#include <span>

#include "forge_rt/device_provider.hpp"

namespace forge_rt::providers {

class CpuProvider final : public DeviceProvider {
public:
    CpuProvider();
    ~CpuProvider() override;

    Result<Buffer> allocate(const BufferSpec& spec) override;
    Result<Event> copy(Buffer& dst, const Buffer& src,
                       std::span<const Event> deps) override;
    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override;
    Result<void> wait(const Event& event) override;

    MemorySpace memory_space() const override { return MemorySpace::Host; }
    TargetKind target() const override { return TargetKind::CPU; }

private:
    std::atomic<uint64_t> next_event_id_{1};
};

}  // namespace forge_rt::providers
