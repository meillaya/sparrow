#pragma once

// MockDeviceProvider — fake DeviceProvider for scheduler tests.
// Lives only in `runtime/tests/`; never linked into the real runtime.
// Records every launch / copy / allocate call in a vector of strings
// so tests can assert that the scheduler dispatched nodes in the
// right topological order. Also folds `events` into a single
// monotonically-incrementing id and marks every produced Event
// complete before returning — same behaviour as the Phase 1 CPU
// stub that real (Phase 11) TritonProvider will eventually expose.
#include "forge_rt/device_provider.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace forge_rt {

class MockDeviceProvider : public DeviceProvider {
public:
    explicit MockDeviceProvider(MemorySpace ms = MemorySpace::Host,
                                TargetKind  tk = TargetKind::CPU)
        : ms_(ms), tk_(tk) {}

    // ------------------------------------------------------------------
    // Inspection — read by the test after each Scheduler::execute.
    // ------------------------------------------------------------------
    const std::vector<std::string>& call_log() const noexcept { return calls_; }

    // Reset the log between sub-tests so each test starts clean.
    void clear_log() noexcept { calls_.clear(); }

    // ------------------------------------------------------------------
    // DeviceProvider interface — all no-ops except for the side-effect
    // of recording into `calls_`.
    // ------------------------------------------------------------------
    Result<Buffer> allocate(const BufferSpec& spec) override {
        calls_.push_back("allocate " + std::to_string(spec.byte_size));
        // Phase-1 stub: return a default Buffer with no data. The
        // scheduler never reads from `allocate`'s result directly —
        // it routes inputs/outputs through the live buffer table —
        // so an empty Buffer is sufficient for the test fixture.
        return Buffer{};
    }

    Result<Event> copy(Buffer& dst, const Buffer& src,
                       std::span<const Event> deps) override {
        calls_.push_back("copy");
        (void)dst; (void)src; (void)deps;
        Event ev{next_id_++};
        ev.mark_completed();
        return ev;
    }

    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override {
        // The scheduler hands us an ArtifactDescriptor whose
        // entry_point is "node <id>" by convention (see scheduler.cpp).
        // We push a "launch <entry_point>" record so the test fixture
        // can pattern-match on the entire string.
        calls_.push_back("launch " + artifact.entry_point);
        (void)inputs; (void)outputs; (void)deps;
        Event ev{next_id_++};
        ev.mark_completed();
        return ev;
    }

    Result<void> wait(const Event& event) override {
        // Phase-1 stub: events are always completed by the time they
        // reach this provider. A real Phase-5 provider would call
        // cudaEventSynchronize / hipEventQuery here.
        if (!event.completed()) {
            return Error{"MockDeviceProvider::wait: event " +
                             std::to_string(event.id()) + " not completed",
                         -1};
        }
        return Result<void>{};
    }

    MemorySpace memory_space() const noexcept override { return ms_; }
    TargetKind  target()       const noexcept override { return tk_; }

private:
    MemorySpace ms_;
    TargetKind  tk_;
    uint64_t    next_id_ = 1;  // 0 is the "default-constructed/no event" sentinel
    std::vector<std::string> calls_;
};

}  // namespace forge_rt
