#pragma once

// ForgeRT C++ ABI — liveness analysis (spec §10, Phase 3).
// A backward dataflow pass over an ExecutionPlan that, for every
// tensor mentioned in `plan.tensors`, computes the [first_use, last_use]
// NodeId interval during which the tensor's backing buffer must stay
// live. The scheduler (Task 33-38) and the buffer-reuse allocator
// (Task 27) consume these intervals; the runtime never recomputes them.
//
// Implementation lives in `runtime/lib/liveness.cpp`; the body is a
// straightforward O(N*M) walk over (node, tensor) pairs and is hot-path
// free at schedule time — analyse is invoked once per plan, not per
// launch.
#include <cstdint>
#include <vector>

#include "forge_rt/execution_plan.hpp"

namespace forge_rt {

// A tensor's live interval, expressed in NodeId coordinates. `first_use`
// is the smallest NodeId that reads or writes the tensor; `last_use` is
// the largest. The interval is inclusive on both ends. Intermediates
// that are written by exactly one node and never read again (DCE
// candidates in a later phase) are still reported — Phase 3 conservatively
// keeps them so a downstream planner pass can decide to drop them.
struct LiveRange {
    TensorId id = 0;
    NodeId   first_use = 0;
    NodeId   last_use  = 0;
};

// Computes a LiveRange for every tensor in `plan.tensors`. The returned
// vector is in plan.tensors order (not sorted by id); callers that need
// a dense id-keyed view should build their own table.
class LivenessAnalyzer {
public:
    static std::vector<LiveRange> analyze(const ExecutionPlan& plan);
};

}  // namespace forge_rt
