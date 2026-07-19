#pragma once

// ForgeRT C++ ABI — MemoryPlanner (spec §10, Phase 3).
// Top-of-the-line orchestration for the buffer-reuse pipeline. Runs
// LivenessAnalyzer over the plan, stamps each LiveRange with its
// byte_size (from plan.tensors), and feeds the result into
// BufferReuseAllocator::allocate. The returned allocations are then
// written into `plan.allocations` so downstream phases (the eventual
// scheduler hook in Task 33-38, the observability memory report in
// Task 44-48) can read them back.
//
// Phase 3 always writes MemorySpace::Host into the AllocationDefinition;
// heterogeneous partitioning (Phase 7) is what promotes tensors to
// Device0/Device1 before this stage runs.
#include <cstdint>

#include "forge_rt/execution_plan.hpp"
#include "forge_rt/result.hpp"

namespace forge_rt {

class MemoryPlanner {
public:
    // Run the planner. On success, mutates `plan.allocations` to
    // contain one AllocationDefinition per tensor with a valid
    // (byte_offset, byte_size, alignment, memory_space) tuple. On
    // failure (e.g. pool too small for the working set), returns an
    // Error and leaves plan.allocations untouched.
    //
    // Defaults: total_bytes = 512 MiB, alignment = 64. These match the
    // Phase 3 reference Triton-CPU kernel's pool sizing.
    static Result<void> plan(ExecutionPlan& plan,
                             uint64_t total_bytes = 512ULL * 1024 * 1024,
                             uint64_t alignment   = 64);
};

}  // namespace forge_rt
