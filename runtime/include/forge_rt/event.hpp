#pragma once

// ForgeRT C++ ABI — Event (spec §10).
// An Event records completion of one asynchronous unit. Copies share the
// same state, so the scheduler can publish one completion to several
// output tensors and a provider can mark it from another thread. The
// completion flag is atomic.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

#include "result.hpp"

namespace forge_rt {

class Event {
public:
    Event() = default;
    explicit Event(uint64_t id) : id_(id), state_(std::make_shared<State>()) {
        state_->completed.store(false, std::memory_order_release);
    }

    bool completed() const noexcept {
        return !state_ || state_->completed.load(std::memory_order_acquire);
    }

    void mark_completed() noexcept {
        if (state_) {
            state_->completed.store(true, std::memory_order_release);
        }
    }

    uint64_t id() const noexcept { return id_; }

    // Block until the event completes, or `timeout` elapses.
    // timeout == 0ms means "poll once and return immediately".
    // Returns Result<void>: ok() iff completed within the budget.
    Result<void> wait(std::chrono::milliseconds timeout = std::chrono::milliseconds{0});

private:
    struct State {
        std::atomic<bool> completed{true};
    };

    uint64_t id_ = 0;
    // Default-constructed events are trivially done. An event built with
    // an id starts incomplete and shares its state across copies.
    std::shared_ptr<State> state_ = std::make_shared<State>();
};

}  // namespace forge_rt
