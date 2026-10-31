#pragma once

// ForgeRT — AstraDeviceProvider (spec §10, §9).
//
// Shipped scope: the "Astra" provider owns the host-side implementation
// of the DeviceProvider interface for the Astra target and delegates
// every `launch()` to the in-process Python simulator at
// ``backends.astra.sim.run_microcode``. The simulator runs on the host
// in a child Python interpreter (the pybind11 binding is a Phase 5
// deliverable), so MemorySpace is Host and every returned Event is
// marked completed on return — Astra is logically synchronous from
// the scheduler's point of view. The DeviceProvider ABI surface is
// locked; this provider adds the Astra target to the registry without
// disturbing the existing Triton / CPU paths.
//
// Spec §20 hard rules honored by this provider:
//   * Does NOT replace the CUDA allocator (we never allocate on Device0).
//   * Does NOT expose the Astra dialect IR — only the compiled
//     microcode (a `vector<byte>` carrying packed 64-bit instruction
//     words) is read back from the artifact.
//   * launch fires `python3 -c <script>` against the project's venv
//     and pipes a JSON spec (input tensors + shape + microcode words)
//     in. stdout carries a base64 payload for the resulting tensor.
#include <atomic>
#include <cstdint>
#include <span>
#include <string>

#include "forge_rt/device_provider.hpp"

namespace forge_rt::providers {

class AstraProvider final : public DeviceProvider {
public:
    // ``python_executable`` overrides the interpreter path. The default
    // is empty, in which case `launch()` falls back to the `python3`
    // on PATH. Tests can pin a specific interpreter (e.g. the
    // `.venv/bin/python`) to avoid PATH drift across hosts.
    explicit AstraProvider(std::string python_executable = {});
    ~AstraProvider() override;

    // Allocate a host-resident buffer honoring `spec.alignment`.
    // Mirrors Triton's allocator: clamped power-of-two alignment,
    // std::aligned_alloc-backed, owns=true.
    Result<Buffer> allocate(const BufferSpec& spec) override;

    // Copy `src` into `dst`. Only Host↔Host; same semantics as Triton.
    Result<Event> copy(Buffer& dst, const Buffer& src,
                       std::span<const Event> deps) override;

    // Launch the compiled artifact. Builds a JSON spec from the
    // input tensors, fires the Python child process, parses the
    // returned tensor payload, and copies it into `outputs[0]`.
    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override;

    // Wait is a no-op today: every Event returned from copy/launch is
    // already marked completed (the Python child process is synchronous).
    Result<void> wait(const Event& event) override;

    MemorySpace memory_space() const override { return MemorySpace::Host; }
    TargetKind  target() const override { return TargetKind::Astra; }

private:
    // Monotonic event-id source (mirrors Triton's pattern).
    std::atomic<uint64_t> next_event_id_{1};

    // Resolved at construction; may be empty, in which case launch()
    // falls back to `python3`. The string is owned by the provider.
    std::string python_executable_;
};

}  // namespace forge_rt::providers
