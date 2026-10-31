#pragma once

// ForgeRT C++ ABI — MemoryPlanner (spec §10, Phase 3).
// Orchestrates the buffer-reuse pipeline. Runs LivenessAnalyzer in
// schedule order, stamps each LiveRange with its byte_size, and feeds
// the internal ranges into BufferReuseAllocator::allocate. External
// inputs and outputs are not placed in the reuse pool: the caller owns
// those buffers, so an output must never share an offset with an input
// whose lifetime has ended.
//
// The returned allocations are written into `plan.allocations`, one row
// per tensor in plan.tensors order. MemorySpace comes from the tensor
// definition, not a hardcoded Host.
#include <cstdint>
#include <span>

#include "forge_rt/execution_plan.hpp"
#include "forge_rt/result.hpp"

namespace forge_rt {

class MemoryPlanner {
public:
    // Run the planner with an explicit schedule order. On success,
    // mutates `plan.allocations` to contain one AllocationDefinition
    // per tensor. External inputs and outputs get offset 0 and their
    // real size; internal tensors get reused offsets. On failure,
    // returns an Error and leaves plan.allocations untouched.
    //
    // Defaults: total_bytes = 512 MiB, alignment = 64.
    static Result<void> plan(ExecutionPlan& plan,
                             std::span<const NodeId> order,
                             uint64_t total_bytes = 512ULL * 1024 * 1024,
                             uint64_t alignment = 64);

    // Convenience overload. Treats plan.nodes order as the schedule
    // order. The scheduler uses the span overload with its topological
    // order; this overload exists for tests and simple callers.
    static Result<void> plan(ExecutionPlan& plan,
                             uint64_t total_bytes = 512ULL * 1024 * 1024,
                             uint64_t alignment = 64);
};

}  // namespace forge_rt
