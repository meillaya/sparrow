#pragma once

// ForgeRT C++ ABI — abstract DeviceProvider (spec §10).
// A DeviceProvider owns allocation, copy, and launch for ONE (memory_space,
// target) pair. The scheduler in Task 12 holds a registry of providers keyed
// by MemorySpace and dispatches to them via this interface.
#include <span>

#include "artifact_descriptor.hpp"
#include "buffer.hpp"
#include "event.hpp"
#include "result.hpp"

namespace forge_rt {

class DeviceProvider {
public:
    virtual ~DeviceProvider() = default;

    // Allocate a buffer with the given spec. Provider must honor
    // spec.alignment (e.g. via posix_memalign or cudaMalloc).
    virtual Result<Buffer> allocate(const BufferSpec& spec) = 0;

    // Copy `src` into `dst`. Returns an Event that fires when the copy is
    // complete (host→host: synchronous; device→device: enqueued). `deps`
    // are upstream events the copy must wait on.
    virtual Result<Event> copy(Buffer& dst, const Buffer& src,
                               std::span<const Event> deps) = 0;

    // Launch the compiled artifact with the given input/output buffers and
    // upstream events. Returns an Event that fires when the launch
    // completes (Phase 1 CPU provider: synchronous; mark_completed before
    // returning).
    virtual Result<Event> launch(const ArtifactDescriptor& artifact,
                                 std::span<const Buffer> inputs,
                                 std::span<Buffer> outputs,
                                 std::span<const Event> deps) = 0;

    // Block until `event` completes.
    virtual Result<void> wait(const Event& event) = 0;

    virtual MemorySpace memory_space() const = 0;
    virtual TargetKind target() const = 0;
};

}  // namespace forge_rt
