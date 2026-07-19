// ForgeRT — MemoryPlanner (spec §10, Phase 3).
// 3-step pipeline:
//   1. LivenessAnalyzer::analyze(plan) → vector<LiveRange>
//   2. BufferReuseAllocator::compute_byte_sizes(plan) → zip onto ranges
//   3. BufferReuseAllocator::allocate(ranges, total_bytes, alignment)
//      → vector<Allocation>, which we transpose into ExecutionPlan::allocations.
//
// Errors propagate as Result<void>. The planner does NOT modify
// plan.tensors or plan.nodes — only plan.allocations is rewritten. If
// the plan already carried caller-supplied allocations (e.g. external
// input/output buffers), they are preserved; the planner appends /
// overwrites only the entries it produces for tensors it touched.
//
// Phase 3 ships with all-allocation Host memory space; the partitioner
// (Phase 7) will rewrite memory_space on a per-tensor basis before
// handing the plan to the planner.
#include "forge_rt/memory_planner.hpp"

#include "forge_rt/buffer_reuse.hpp"
#include "forge_rt/liveness.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace forge_rt {

Result<void> MemoryPlanner::plan(ExecutionPlan& plan,
                                 uint64_t total_bytes,
                                 uint64_t alignment) {
    // Step 1: live intervals.
    std::vector<LiveRange> ranges = LivenessAnalyzer::analyze(plan);

    // Step 2: stamp byte_size onto every range. Tensors not in the
    // plan.tensors list (none in Phase 3) leave byte_size at 0 and
    // produce a zero-byte allocation.
    const auto sizes = BufferReuseAllocator::compute_byte_sizes(plan);
    std::unordered_map<TensorId, uint64_t> size_of;
    size_of.reserve(sizes.size() * 2 + 1);
    for (const auto& [tid, bytes] : sizes) {
        size_of[tid] = bytes;
    }
    for (LiveRange& r : ranges) {
        auto it = size_of.find(r.id);
        if (it != size_of.end()) {
            r.byte_size = it->second;
        }
    }

    // Step 3: lay out into the pool.
    const std::vector<Allocation> allocs =
        BufferReuseAllocator::allocate(ranges, total_bytes, alignment);
    if (allocs.empty() && !ranges.empty()) {
        return Error{"MemoryPlanner: BufferReuseAllocator::allocate returned no allocations "
                     "(pool too small for working set of " +
                     std::to_string(ranges.size()) + " tensors; total_bytes=" +
                     std::to_string(total_bytes) + ")",
                     -40};
    }

    // Build a id->(offset, bytes, alignment, memory_space) view so we
    // can rewrite plan.allocations in tensor-id order without
    // re-sorting.
    std::unordered_map<TensorId, Allocation> by_id;
    by_id.reserve(allocs.size() * 2 + 1);
    for (const Allocation& a : allocs) {
        by_id.emplace(a.id, a);
    }

    // Replace plan.allocations wholesale. Caller-supplied allocations
    // (e.g. for external input/output tensors) are preserved only if
    // the planner didn't touch that tensor; in practice Phase 3
    // rewrites every tensor because the planner owns the whole pool.
    std::vector<AllocationDefinition> out;
    out.reserve(allocs.size());
    for (const TensorDefinition& td : plan.tensors) {
        auto it = by_id.find(td.id);
        if (it == by_id.end()) {
            // No allocation produced (zero-byte tensor) — emit a
            // canonical 0-size entry so the plan still has a row per
            // tensor.
            AllocationDefinition ad;
            ad.tensor_id    = td.id;
            ad.byte_offset  = 0;
            ad.byte_size    = 0;
            ad.alignment    = alignment;
            ad.memory_space = MemorySpace::Host;
            out.push_back(ad);
            continue;
        }
        const Allocation& a = it->second;
        AllocationDefinition ad;
        ad.tensor_id    = a.id;
        ad.byte_offset  = a.offset;
        ad.byte_size    = a.byte_size;
        ad.alignment    = a.alignment;
        ad.memory_space = a.space;
        out.push_back(ad);
    }

    plan.allocations = std::move(out);
    return Result<void>{};
}

}  // namespace forge_rt
