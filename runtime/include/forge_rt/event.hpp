#pragma once

// ForgeRT C++ ABI — Event (spec §10).
// Phase 1: CPU stub. The Event records "is this work item done?" and supports
// `wait()` with a timeout. Phase 5 will replace this with real fences backed
// by cudaEvent_t / HIP event / HLS stream sync. The ABI surface is locked —
// only the implementation moves.
#include <cstdint>
#include <chrono>
#include "result.hpp"

namespace forge_rt {

class Event {
public:
    Event() = default;
    explicit Event(uint64_t id) : id_(id), completed_(false) {}

    bool completed() const noexcept { return completed_; }
    void mark_completed() noexcept { completed_ = true; }
    uint64_t id() const noexcept { return id_; }

    // Block until `completed_` becomes true, or `timeout` elapses.
    // timeout == 0ms means "poll once and return immediately".
    // Returns Result<void>: ok() iff completed within the budget.
    Result<void> wait(std::chrono::milliseconds timeout = std::chrono::milliseconds{0});

private:
    uint64_t id_ = 0;
    bool completed_ = true;  // default-constructed Events are trivially "done"
};

}  // namespace forge_rt
