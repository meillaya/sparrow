#pragma once

// ForgeRT — MicroGpuDeviceProvider (spec §11).
//
// Shipped scope: the "MicroGPU" provider owns the host-side
// implementation of the DeviceProvider interface for the MicroGPU
// target (Phase 8). Every `launch()` delegates to the in-process
// Python reference interpreter at
// ``backends.microgpu.interp.run`` (loaded via subprocess to keep
// the C++ side ABI-stable and to match the astra provider pattern).
// The interpreter runs on the host in a child Python interpreter, so
// MemorySpace is Host and every returned Event is marked completed
// on return — MicroGPU is logically synchronous from the scheduler's
// point of view. The DeviceProvider ABI surface is locked; this
// provider adds the MicroGPU target to the registry without
// disturbing the existing Triton / Astra paths.
//
// Spec §20 hard rules honored by this provider:
//   * Does NOT replace the CUDA allocator (we never allocate on
//     Device0 — MicroGPU is Host-only by construction).
//   * Does NOT expose the MicroGPU dialect IR — only the compiled
//     bytecode (a `vector<byte>` carrying packed 16-bit instruction
//     words) is read back from the artifact.
//   * launch fires `python3 -c <script>` against the project's venv
//     and pipes a JSON spec (input tensors + bytecode words) in.
//     stdout carries a base64 payload for the resulting tensor.

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "forge_rt/device_provider.hpp"

namespace forge_rt::providers {

class MicroGpuProvider final : public DeviceProvider {
public:
    // ``python_executable`` overrides the interpreter path. The
    // default is empty, in which case `launch()` falls back to the
    // `python3` on PATH. Tests can pin a specific interpreter
    // (e.g. the `.venv/bin/python`) to avoid PATH drift across hosts.
    explicit MicroGpuProvider(std::string python_executable = {});
    ~MicroGpuProvider() override;

    // Allocate a host-resident buffer honoring `spec.alignment`.
    // Mirrors Triton's allocator: clamped power-of-two alignment,
    // std::aligned_alloc-backed, owns=true.
    Result<Buffer> allocate(const BufferSpec& spec) override;

    // Copy `src` into `dst`. Only Host<->Host; same semantics as
    // Triton / Astra.
    Result<Event> copy(Buffer& dst, const Buffer& src,
                       std::span<const Event> deps) override;

    // Launch the compiled artifact. Builds a JSON spec from the
    // input tensors + artifact.binary (the packed 16-bit bytecode
    // words), fires the Python child process, parses the returned
    // tensor payload, and copies it into `outputs[0]`.
    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override;

    // Wait is a no-op today: every Event returned from copy/launch
    // is already marked completed (the Python child process is
    // synchronous).
    Result<void> wait(const Event& event) override;

    MemorySpace memory_space() const override { return MemorySpace::Host; }
    TargetKind  target() const override { return TargetKind::MicroGPU; }

private:
    // Monotonic event-id source (mirrors Triton's pattern).
    std::atomic<uint64_t> next_event_id_{1};

    // Resolved at construction; may be empty, in which case launch()
    // falls back to `python3`. The string is owned by the provider.
    std::string python_executable_;
};

}  // namespace forge_rt::providers