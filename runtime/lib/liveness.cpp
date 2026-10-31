// ForgeRT — LivenessAnalyzer (spec §10, Phase 3).
// Per-tensor interval over an ExecutionPlan. For each tensor we walk the
// plan's nodes in schedule order and record the first and last schedule
// position that mentions the tensor in `inputs` or `outputs`. Positions
// come from the scheduler's topological order, not from the raw node
// ids, so two tensors that are live at the same time cannot receive
// disjoint intervals just because their producer ids are far apart.
//
// The result is a LiveRange vector aligned 1:1 with `plan.tensors`.
#include "forge_rt/liveness.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace forge_rt {

namespace {

constexpr uint32_t kNoUse = std::numeric_limits<uint32_t>::max();

}  // namespace

std::vector<LiveRange> LivenessAnalyzer::analyze(const ExecutionPlan& plan) {
    std::vector<NodeId> order;
    order.reserve(plan.nodes.size());
    for (const ExecutableNode& node : plan.nodes) {
        order.push_back(node.id);
    }
    return analyze(plan, order);
}

std::vector<LiveRange> LivenessAnalyzer::analyze(const ExecutionPlan& plan,
                                                 std::span<const NodeId> order) {
    std::vector<LiveRange> out;
    out.reserve(plan.tensors.size());

    // Initialise every range with a sentinel for first_use so we can
    // `min` into it without branching. last_use stays at 0; a tensor
    // mentioned only at position 0 collapses to a point interval.
    for (const TensorDefinition& td : plan.tensors) {
        LiveRange range;
        range.id = td.id;
        range.first_use = kNoUse;
        range.last_use = 0;
        range.space = td.memory_space;
        out.push_back(range);
    }

    // TensorId -> index in plan.tensors.
    std::unordered_map<TensorId, std::size_t> index_of;
    index_of.reserve(plan.tensors.size() * 2 + 1);
    for (std::size_t i = 0; i < plan.tensors.size(); ++i) {
        index_of.emplace(plan.tensors[i].id, i);
    }

    // NodeId -> schedule position.
    std::unordered_map<NodeId, uint32_t> position_of;
    position_of.reserve(order.size() * 2 + 1);
    for (std::size_t position = 0; position < order.size(); ++position) {
        position_of.emplace(order[position], static_cast<uint32_t>(position));
    }

    const auto record = [&](TensorId tid, uint32_t position) {
        const auto it = index_of.find(tid);
        if (it == index_of.end()) {
            return;
        }
        LiveRange& range = out[it->second];
        range.first_use = std::min(range.first_use, position);
        range.last_use = std::max(range.last_use, position);
    };

    for (const ExecutableNode& node : plan.nodes) {
        const auto position_it = position_of.find(node.id);
        if (position_it == position_of.end()) {
            continue;
        }
        const uint32_t position = position_it->second;
        for (const TensorId tid : node.inputs) {
            record(tid, position);
        }
        for (const TensorId tid : node.outputs) {
            record(tid, position);
        }
    }

    // Collapse the "not seen" sentinel to a point interval.
    for (LiveRange& range : out) {
        if (range.first_use == kNoUse) {
            range.first_use = 0;
            range.last_use = 0;
        }
    }

    return out;
}

}  // namespace forge_rt
