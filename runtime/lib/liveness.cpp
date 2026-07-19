// ForgeRT — LivenessAnalyzer (spec §10, Phase 3).
// Per-tensor backward dataflow interval over an ExecutionPlan. For each
// tensor we walk the plan's node DAG once and record the smallest /
// largest NodeId over every node that has the tensor in `inputs` or
// `outputs`. The result is a LiveRange vector aligned 1:1 with
// `plan.tensors` so callers can index by position or build a TensorId
// lookup table on top.
//
// We use UINT32_MAX as a "not yet seen" sentinel so the early-exit
// stays branch-free in the inner loop: a single `if (new_first < ...)`
// beats an extra `else` and keeps the O(N*M) walk tight. Tensors that
// are declared but never referenced (e.g. pre-allocated constants the
// caller will install by PlanInput) collapse to first_use == last_use
// == 0, which the caller can interpret as "synthetic; ignore".
#include "forge_rt/liveness.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace forge_rt {

std::vector<LiveRange> LivenessAnalyzer::analyze(const ExecutionPlan& plan) {
    std::vector<LiveRange> out;
    out.reserve(plan.tensors.size());

    // First pass: initialise ranges with a sentinel for first_use so we
    // can `min` into them without branching. last_use stays at 0 — the
    // worst case is a tensor only mentioned at node 0, in which case
    // first_use == last_use == 0 and the interval collapses to a point.
    constexpr NodeId kNoUse = std::numeric_limits<NodeId>::max();
    for (const TensorDefinition& td : plan.tensors) {
        LiveRange r;
        r.id        = td.id;
        r.first_use = kNoUse;
        r.last_use  = 0;
        out.push_back(r);
    }

    // Build a TensorId -> index map so the inner loop is O(1) per
    // tensor reference rather than an O(M) scan. Phase 1 keeps plan
    // tensors in id-ascending order so we could skip the map, but a
    // real compiler emits them in dependency order — the map is cheap
    // and keeps us honest for Task 33+.
    std::vector<std::pair<TensorId, std::size_t>> id_index;
    id_index.reserve(plan.tensors.size());
    for (std::size_t i = 0; i < plan.tensors.size(); ++i) {
        id_index.emplace_back(plan.tensors[i].id, i);
    }
    std::sort(id_index.begin(), id_index.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    auto index_of = [&](TensorId tid) -> std::size_t {
        auto it = std::lower_bound(id_index.begin(), id_index.end(), tid,
            [](const auto& p, TensorId v) { return p.first < v; });
        if (it == id_index.end() || it->first != tid) return out.size();
        return it->second;
    };

    // Walk every (node, input) and (node, output) pair and fold the
    // extremes into the matching LiveRange. O(N*M) where N is node
    // count and M is total I/O slots — Phase 3 plans stay small.
    for (const ExecutableNode& node : plan.nodes) {
        for (TensorId tid : node.inputs) {
            const std::size_t idx = index_of(tid);
            if (idx >= out.size()) continue;  // tensor not declared in plan
            if (node.id < out[idx].first_use) out[idx].first_use = node.id;
            if (node.id > out[idx].last_use)  out[idx].last_use  = node.id;
        }
        for (TensorId tid : node.outputs) {
            const std::size_t idx = index_of(tid);
            if (idx >= out.size()) continue;
            if (node.id < out[idx].first_use) out[idx].first_use = node.id;
            if (node.id > out[idx].last_use)  out[idx].last_use  = node.id;
        }
    }

    // Collapse the "not seen" sentinel back to 0 so callers see a
    // well-defined point interval for unused tensors. This matches the
    // convention that a tensor with no node references still has a
    // zero-length live range rather than a poison value.
    for (LiveRange& r : out) {
        if (r.first_use == kNoUse) {
            r.first_use = 0;
        }
    }

    return out;
}

}  // namespace forge_rt
