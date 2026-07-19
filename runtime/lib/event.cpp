// ForgeRT C++ ABI — Event (spec §10, CPU stub).
#include "forge_rt/event.hpp"

#include <chrono>
#include <thread>

namespace forge_rt {

Result<void> Event::wait(std::chrono::milliseconds timeout) {
    if (completed_) return Result<void>{};

    if (timeout == std::chrono::milliseconds{0}) {
        return Error{"Event " + std::to_string(id_) + " not completed (timeout=0ms)", -1};
    }

    // CPU stub: spin with a 1ms sleep until the deadline expires. The
    // API surface is locked and the stub lets the scheduler + provider
    // integration exercise end-to-end without GPU/HW dependencies.
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + timeout;
    while (!completed_) {
        if (clock::now() >= deadline) {
            return Error{"Event " + std::to_string(id_) + " timed out after " +
                             std::to_string(timeout.count()) + "ms",
                         -2};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return Result<void>{};
}

}  // namespace forge_rt
