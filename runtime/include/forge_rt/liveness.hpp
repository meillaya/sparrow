#pragma once

// ForgeRT C++ ABI — liveness analysis (spec §10, Phase 3).
// A backward dataflow pass over an ExecutionPlan that, for every
// tensor mentioned in `plan.tensors`, computes the [first_use, last_use]
// interval during which the tensor's backing buffer must stay live.
// Intervals are expressed in schedule positions, not raw node ids, so a
// plan whose ids are not topologically sorted still gets a sound
// buffer-reuse layout.
//
// Implementation lives in `runtime/lib/liveness.cpp`; the body is a
// straightforward O(N*M) walk over (node, tensor) pairs and is hot-path
// free at schedule time: analysis runs once per plan, not per launch.
#include <cstdint>
#include <span>
#include <vector>

#include "forge_rt/execution_plan.hpp"

namespace forge_rt {

// A tensor's live interval, expressed in schedule positions. `first_use`
// is the earliest position that reads or writes the tensor; `last_use`
// is the latest. The interval is inclusive on both ends.
//
// `byte_size` is the planner footprint: numel * kByteSize(dtype), with
// Int4's packing applied. LivenessAnalyzer::analyze leaves it at 0; the
// BufferReuseAllocator / MemoryPlanner stamp it from compute_byte_sizes
// before sorting.
struct LiveRange {
    TensorId id = 0;
    uint32_t first_use = 0;
    uint32_t last_use = 0;
    uint64_t byte_size = 0;
    MemorySpace space = MemorySpace::Host;
};

// Computes a LiveRange for every tensor in `plan.tensors`. The returned
// vector is in plan.tensors order (not sorted by id).
//
// The single-argument overload treats `plan.nodes` order as the schedule
// order. It exists for tests and for callers that already emit nodes in
// topological order. The scheduler calls the span overload with the
// order returned by its topological sort.
class LivenessAnalyzer {
public:
    static std::vector<LiveRange> analyze(const ExecutionPlan& plan);
    static std::vector<LiveRange> analyze(const ExecutionPlan& plan,
                                          std::span<const NodeId> order);
};

}  // namespace forge_rt
