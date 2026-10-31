// ForgeRT — MemoryPlanner (spec §10, Phase 3).
// Pipeline:
//   1. LivenessAnalyzer::analyze(plan, order) -> vector<LiveRange>
//   2. BufferReuseAllocator::compute_byte_sizes(plan) -> zip onto ranges
//   3. BufferReuseAllocator::allocate(internal ranges, ...) -> vector<Allocation>
//   4. Write one AllocationDefinition per tensor, in plan.tensors order.
//
// External inputs and outputs are excluded from the reuse pool. Their
// allocations carry offset 0 and their real size, and the scheduler
// aliases the caller's buffers instead of allocating. This prevents an
// output from reusing an input's offset after the input's last use.
//
// Errors propagate as Result<void>. The planner does not modify
// plan.tensors or plan.nodes, and on error leaves plan.allocations
// untouched.
#include "forge_rt/memory_planner.hpp"

#include "forge_rt/buffer_reuse.hpp"
#include "forge_rt/liveness.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace forge_rt {

namespace {

std::span<const NodeId> plan_order(const ExecutionPlan& plan,
                                   std::vector<NodeId>& storage) {
    storage.reserve(plan.nodes.size());
    for (const ExecutableNode& node : plan.nodes) {
        storage.push_back(node.id);
    }
    return storage;
}

}  // namespace

Result<void> MemoryPlanner::plan(ExecutionPlan& plan,
                                 uint64_t total_bytes,
                                 uint64_t alignment) {
    std::vector<NodeId> storage;
    const std::span<const NodeId> order = plan_order(plan, storage);
    return MemoryPlanner::plan(plan, order, total_bytes, alignment);
}

Result<void> MemoryPlanner::plan(ExecutionPlan& plan,
                                 std::span<const NodeId> order,
                                 uint64_t total_bytes,
                                 uint64_t alignment) {
    if (alignment == 0) {
        return Error{"MemoryPlanner: alignment must be greater than zero", -41};
    }

    std::vector<LiveRange> ranges = LivenessAnalyzer::analyze(plan, order);

    const auto sizes = BufferReuseAllocator::compute_byte_sizes(plan);
    std::unordered_map<TensorId, uint64_t> size_of;
    size_of.reserve(sizes.size() * 2 + 1);
    for (const auto& [tid, bytes] : sizes) {
        size_of[tid] = bytes;
    }

    std::unordered_map<TensorId, MemorySpace> space_of;
    space_of.reserve(plan.tensors.size() * 2 + 1);
    for (const TensorDefinition& tensor : plan.tensors) {
        space_of[tensor.id] = tensor.memory_space;
    }

    for (LiveRange& range : ranges) {
        const auto size_it = size_of.find(range.id);
        if (size_it != size_of.end()) {
            range.byte_size = size_it->second;
        }
        const auto space_it = space_of.find(range.id);
        if (space_it != space_of.end()) {
            range.space = space_it->second;
        }
    }

    std::unordered_set<TensorId> external;
    external.reserve(plan.inputs.size() + plan.outputs.size());
    for (const PlanInput& input : plan.inputs) {
        external.insert(input.tensor_id);
    }
    for (const PlanOutput& output : plan.outputs) {
        external.insert(output.tensor_id);
    }

    std::vector<LiveRange> internal_ranges;
    internal_ranges.reserve(ranges.size());
    for (const LiveRange& range : ranges) {
        if (external.contains(range.id)) {
            continue;
        }
        internal_ranges.push_back(range);
    }

    const std::vector<Allocation> allocations =
        BufferReuseAllocator::allocate(internal_ranges, total_bytes, alignment);
    if (allocations.empty() && !internal_ranges.empty()) {
        return Error{"MemoryPlanner: BufferReuseAllocator::allocate returned no "
                     "allocations (pool too small for working set of " +
                         std::to_string(internal_ranges.size()) +
                         " tensors; total_bytes=" + std::to_string(total_bytes) + ")",
                     -40};
    }

    std::unordered_map<TensorId, Allocation> by_id;
    by_id.reserve(allocations.size() * 2 + 1);
    for (const Allocation& allocation : allocations) {
        by_id.emplace(allocation.id, allocation);
    }

    std::vector<AllocationDefinition> out;
    out.reserve(plan.tensors.size());
    for (const TensorDefinition& tensor : plan.tensors) {
        const uint64_t bytes = size_of.contains(tensor.id) ? size_of[tensor.id] : 0;
        if (external.contains(tensor.id)) {
            AllocationDefinition row;
            row.tensor_id = tensor.id;
            row.byte_offset = 0;
            row.byte_size = bytes;
            row.alignment = alignment;
            row.memory_space = tensor.memory_space;
            out.push_back(row);
            continue;
        }
        const auto allocation_it = by_id.find(tensor.id);
        if (allocation_it == by_id.end()) {
            AllocationDefinition row;
            row.tensor_id = tensor.id;
            row.byte_offset = 0;
            row.byte_size = 0;
            row.alignment = alignment;
            row.memory_space = tensor.memory_space;
            out.push_back(row);
            continue;
        }
        const Allocation& allocation = allocation_it->second;
        AllocationDefinition row;
        row.tensor_id = allocation.id;
        row.byte_offset = allocation.offset;
        row.byte_size = allocation.byte_size;
        row.alignment = allocation.alignment;
        row.memory_space = allocation.space;
        out.push_back(row);
    }

    plan.allocations = std::move(out);
    return Result<void>{};
}

}  // namespace forge_rt
