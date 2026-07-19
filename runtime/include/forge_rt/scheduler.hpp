#pragma once

// ForgeRT C++ ABI — topological scheduler (spec §10).
// Owns a registry of DeviceProviders (one per memory space) and walks
// an ExecutionPlan in topological order, dispatching each node to the
// matching provider. Builds an internal TensorId → Buffer "live
// buffer table" and a TensorId → Event "ready event table" so that
// downstream launches can wait on upstream outputs.
//
// Phase 1: only TritonProvider (Host memory space). The scheduler
// itself is provider-agnostic — tests drive it via the
// runtime/tests/mock_provider.hpp fake.
#include <memory>
#include <vector>

#include "buffer.hpp"
#include "device_provider.hpp"
#include "event.hpp"
#include "execution_plan.hpp"
#include "result.hpp"

namespace forge_rt {

class Scheduler {
public:
    // Takes ownership of the providers. The vector order is the
    // device_id → provider mapping: providers_[plan.nodes[i].device]
    // dispatches the i-th node. Future Tasks can let providers
    // self-register by MemorySpace, but for Phase 1 an explicit
    // vector is the simplest contract.
    explicit Scheduler(std::vector<std::shared_ptr<DeviceProvider>> providers);

    // Walk `plan` to completion. `inputs` and `outputs` are passed
    // by reference (not by value) because Buffer is move-only and we
    // need to swap into the live buffer table in place.
    //
    // Returns a vector of Events, one per PlanOutput, in plan.outputs
    // order. Callers can wait on them individually or chain them
    // into downstream ForgeRT runs.
    Result<std::vector<Event>> execute(const ExecutionPlan& plan,
                                       std::vector<Buffer>& inputs,
                                       std::vector<Buffer>& outputs);

private:
    std::vector<std::shared_ptr<DeviceProvider>> providers_;
};

}  // namespace forge_rt
