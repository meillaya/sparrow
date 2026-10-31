#pragma once

// ForgeRT — AstraDeviceProvider (spec §10, §9).
//
// Shipped scope: the "Astra" provider executes a self-contained binary
// artifact on the host. `ArtifactDescriptor.binary` is a JSON header
// with packed 64-bit instruction words and buffer bindings; launch
// copies the plan inputs into a simulated DRAM image, runs the words
// against a 512 KiB SRAM image, and writes the bound output region
// back. There is no child process. MemorySpace is Host and events are
// completed on return, so Astra is logically synchronous.
//
// Spec §20 hard rules honored by this provider:
//   * Does NOT replace the CUDA allocator (we never allocate on Device0).
//   * Does NOT expose the Astra dialect IR — only the compiled
//     microcode words and their buffer bindings are read.
#include <atomic>
#include <cstdint>
#include <span>

#include "forge_rt/device_provider.hpp"

namespace forge_rt::providers {

class AstraProvider final : public DeviceProvider {
public:
    AstraProvider();
    ~AstraProvider() override;

    // Allocate a host-resident buffer honoring `spec.alignment`.
    // Mirrors Triton's allocator: clamped power-of-two alignment,
    // std::aligned_alloc-backed, owns=true.
    Result<Buffer> allocate(const BufferSpec& spec) override;

    // Copy `src` into `dst`. Only Host↔Host; same semantics as Triton.
    Result<Event> copy(Buffer& dst, const Buffer& src,
                       std::span<const Event> deps) override;

    // Launch the compiled artifact: bind inputs, execute words, copy
    // the bound output region into `outputs[0]`.
    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override;

    // Wait is a no-op: every Event returned from copy/launch is already
    // marked completed (execution is synchronous).
    Result<void> wait(const Event& event) override;

    MemorySpace memory_space() const override { return MemorySpace::Host; }
    TargetKind  target() const override { return TargetKind::Astra; }

private:
    // Monotonic event-id source (mirrors Triton's pattern).
    std::atomic<uint64_t> next_event_id_{1};
};

}  // namespace forge_rt::providers
