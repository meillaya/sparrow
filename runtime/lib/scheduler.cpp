// ForgeRT — topological scheduler (spec §10).
// Holds a vector of DeviceProviders (indexed by ExecutableNode::device)
// and walks an ExecutionPlan in Kahn's topological order. For each
// node in topo order: gather the inputs (a slice of the live buffer
// table) and the upstream events (a slice of the live event table),
// dispatch to the matching provider via DeviceProvider::launch, and
// publish the resulting event into the table for downstream nodes.
//
// Implementation notes:
//
// * We use Kahn's algorithm with a NodeId-indexed in-degree table.
//   The "reverse" edges (depender → dependees) are precomputed so
//   the inner loop is O(degree) without any hash lookups.
//
// * The two dense tables (TensorId → Buffer, TensorId → Event) are
//   sized once from plan.tensors.size() so the hot path is just
//   `buffer_table[tid]` / `event_table[tid]` — no hash lookup.
//
// * Per-node I/O buffers are moved into a temporary std::vector<Buffer>
//   for the duration of `provider->launch` (because std::span needs a
//   contiguous range of T, and the table is non-contiguous). They
//   are moved back into the table after the launch, so downstream
//   nodes see the same in-place mutation the provider made. For
//   Phase 1 host stub the buffers are essentially handle-swaps, so
//   the extra move is free. Phase 5 keeps the same code; the only
//   thing that changes is what "Buffer" carries (CUDA pointer).
#include "forge_rt/scheduler.hpp"

#include "forge_rt/abi_types.hpp"
#include "forge_rt/artifact_descriptor.hpp"

#include <chrono>
#include <deque>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace forge_rt {

Scheduler::Scheduler(std::vector<std::shared_ptr<DeviceProvider>> providers)
    : providers_(std::move(providers)) {}

// ---------------------------------------------------------------------------
// Internal helpers — file-local so we don't widen the public surface.
// ---------------------------------------------------------------------------
namespace {

inline std::size_t table_size(const ExecutionPlan& plan) {
    // Dense-vector sizing. We deliberately use the plan's published
    // count rather than the max id so callers can't blow up the
    // table with a sparse TensorId namespace; range checks happen at
    // every access below.
    const std::size_t t = plan.tensors.size();
    const std::size_t n = plan.nodes.size();
    return (t > n ? t : n) + 1;
}

// Sentinel Error used by the topological-sort failure path. The
// `code` (-20) is negative so callers can recognise it as "the plan
// itself is invalid" rather than "the runtime tripped".
Error cycle_error() {
    return Error{"topological sort failed: cycle detected (or unsatisfied dependency)",
                 -20};
}

}  // namespace

Result<std::vector<Event>> Scheduler::execute(const ExecutionPlan& plan,
                                              std::vector<Buffer>& inputs,
                                              std::vector<Buffer>& outputs) {
    // 1. Dense tables, indexed by TensorId / NodeId. We size them
    //    once and rely on bound checks at every access.
    std::vector<Buffer> buffer_table(table_size(plan));
    std::vector<Event>  event_table(table_size(plan));

    // 2. Move caller's PlanInput buffers into the table. After this,
    //    the caller's `inputs` vector holds default Buffers — same
    //    contract every move-only API in this codebase obeys.
    if (plan.inputs.size() != inputs.size()) {
        return Error{"plan.inputs has " + std::to_string(plan.inputs.size()) +
                         " entries but caller passed " + std::to_string(inputs.size()),
                     -11};
    }
    for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
        const TensorId tid = plan.inputs[i].tensor_id;
        if (tid >= buffer_table.size()) {
            return Error{"plan.inputs[" + std::to_string(i) +
                             "] tensor_id out of range",
                         -12};
        }
        buffer_table[tid] = std::move(inputs[i]);
    }

    // 3. Move caller's PlanOutput buffers into the table. They sit
    //    here for the duration of the run and we move them back at
    //    the end — symmetric with the input install.
    if (plan.outputs.size() != outputs.size()) {
        return Error{"plan.outputs has " + std::to_string(plan.outputs.size()) +
                         " entries but caller passed " +
                         std::to_string(outputs.size()),
                     -13};
    }
    for (std::size_t i = 0; i < plan.outputs.size(); ++i) {
        const TensorId tid = plan.outputs[i].tensor_id;
        if (tid >= buffer_table.size()) {
            return Error{"plan.outputs[" + std::to_string(i) +
                             "] tensor_id out of range",
                         -14};
        }
        buffer_table[tid] = std::move(outputs[i]);
    }

    // 4. Inputs are trivially ready (no producer runs before them).
    //    Default-constructed Events are already marked "completed"
    //    (see Event ABI), but we set them explicitly here so the
    //    intent is visible in code review.
    for (const auto& pi : plan.inputs) {
        if (pi.tensor_id < event_table.size()) {
            event_table[pi.tensor_id].mark_completed();
        }
    }

    // 5. Build the reverse-edge map (depender → dependees) and the
    //    NodeId → plan.nodes index. Kahn's algorithm walks the
    //    reverse edges; the index lets us translate NodeId to the
    //    concrete ExecutableNode struct in O(1).
    std::vector<std::vector<NodeId>> successors(table_size(plan));
    std::vector<std::size_t> in_degree(table_size(plan), 0);
    std::unordered_map<NodeId, std::size_t> node_index;
    node_index.reserve(plan.nodes.size() * 2 + 1);
    for (std::size_t i = 0; i < plan.nodes.size(); ++i) {
        const NodeId nid = plan.nodes[i].id;
        node_index[nid] = i;
        for (NodeId dep : plan.nodes[i].dependencies) {
            if (dep >= successors.size()) {
                return Error{"node " + std::to_string(nid) +
                                 " depends on out-of-range node " +
                                 std::to_string(dep),
                             -15};
            }
            successors[dep].push_back(nid);
            in_degree[nid]++;
        }
    }

    // 6. Kahn's algorithm. We seed the ready queue with every
    //    in-degree-0 node (in plan.nodes order for determinism).
    std::deque<NodeId> ready;
    for (std::size_t i = 0; i < plan.nodes.size(); ++i) {
        const NodeId nid = plan.nodes[i].id;
        if (in_degree[nid] == 0) {
            ready.push_back(nid);
        }
    }

    std::vector<NodeId> topo;
    topo.reserve(plan.nodes.size());
    while (!ready.empty()) {
        const NodeId nid = ready.front();
        ready.pop_front();
        topo.push_back(nid);
        for (NodeId succ : successors[nid]) {
            if (in_degree[succ] == 0) {
                // Already processed — duplicates in the successor
                // list are harmless but worth a guard.
                continue;
            }
            if (--in_degree[succ] == 0) {
                ready.push_back(succ);
            }
        }
    }

    if (topo.size() != plan.nodes.size()) {
        return cycle_error();
    }

    // 7. Walk topo order, dispatching each node.
    //
    //    The provider's launch() takes std::span<const Buffer>/Buffer
    //    which require a contiguous range. The buffer table is
    //    dense but indexed by TensorId, so we move the relevant
    //    rows into a temporary vector, span over it, launch, then
    //    move them back. For Phase 1 (handle-only Buffer) the moves
    //    are no-ops; for Phase 5 they're pointer swaps.
    for (NodeId nid : topo) {
        auto idx_it = node_index.find(nid);
        if (idx_it == node_index.end()) {
            return Error{"topological order references unknown node " +
                             std::to_string(nid),
                         -21};
        }
        const ExecutableNode& node = plan.nodes[idx_it->second];

        if (node.device >= providers_.size()) {
            return Error{"node " + std::to_string(node.id) +
                             " references device " +
                             std::to_string(node.device) +
                             " but only " + std::to_string(providers_.size()) +
                             " providers are registered",
                         -22};
        }
        const std::shared_ptr<DeviceProvider>& provider = providers_[node.device];

        // Stage A — move inputs out of the table into a contiguous vec.
        std::vector<Buffer>   in_bufs;
        std::vector<Event>    in_events;
        std::vector<TensorId> in_ids;  // remember where to put them back
        in_bufs.reserve(node.inputs.size());
        in_events.reserve(node.inputs.size());
        in_ids.reserve(node.inputs.size());
        for (TensorId tid : node.inputs) {
            if (tid >= buffer_table.size()) {
                return Error{"node " + std::to_string(node.id) +
                                 " has out-of-range input tensor " +
                                 std::to_string(tid),
                             -23};
            }
            in_ids.push_back(tid);
            in_bufs.push_back(std::move(buffer_table[tid]));
            if (tid < event_table.size()) {
                in_events.push_back(event_table[tid]);
            }
        }

        // Stage B — move outputs out. The provider fills them in
        // place; we put them back after launch so downstream nodes
        // observe the mutation.
        std::vector<Buffer>   out_bufs;
        std::vector<TensorId> out_ids;
        out_bufs.reserve(node.outputs.size());
        out_ids.reserve(node.outputs.size());
        for (TensorId tid : node.outputs) {
            if (tid >= buffer_table.size()) {
                return Error{"node " + std::to_string(node.id) +
                                 " has out-of-range output tensor " +
                                 std::to_string(tid),
                             -24};
            }
            out_ids.push_back(tid);
            out_bufs.push_back(std::move(buffer_table[tid]));
        }

        // Phase 5 will replace this stub. For now the launch entry
        // point is just "node <id>" so the MockProvider can log it
        // verbatim and a real TritonProvider can map it to a cached
        // compilation. TargetKind comes from the provider so the
        // dispatch table doesn't need a redundant plan field.
        ArtifactDescriptor stub;
        stub.target = provider->target();
        stub.entry_point = "node " + std::to_string(node.id);

        const std::span<const Buffer> in_span{in_bufs};
        const std::span<Buffer>       out_span{out_bufs};
        const std::span<const Event>  dep_span{in_events};

        auto launch_res = provider->launch(stub, in_span, out_span, dep_span);
        if (!launch_res.ok()) {
            return Error{"launch failed for node " + std::to_string(node.id) +
                             ": " + launch_res.error().message,
                         -30};
        }
        Event produced = std::move(launch_res.value());

        // All outputs become ready when the kernel completes. In
        // Phase 1 we have one Event for the whole launch; Phase 5
        // will split this into per-output fences (one Event per
        // output buffer) once the real provider returns.
        for (TensorId tid : out_ids) {
            event_table[tid] = produced;
        }

        // Stage C — put buffers back into the live table. Inputs
        // come back unchanged; outputs come back mutated.
        for (std::size_t i = 0; i < in_ids.size(); ++i) {
            buffer_table[in_ids[i]] = std::move(in_bufs[i]);
        }
        for (std::size_t i = 0; i < out_ids.size(); ++i) {
            buffer_table[out_ids[i]] = std::move(out_bufs[i]);
        }
    }

    // 8. Build the event list to return — one Event per PlanOutput,
    //    in plan.outputs order. We do NOT block on them here;
    //    callers can wait themselves (or chain into downstream
    //    ForgeRT runs). For Phase 1's host stub the events are
    //    already complete so wait() returns instantly.
    std::vector<Event> output_events;
    output_events.reserve(plan.outputs.size());
    for (const auto& po : plan.outputs) {
        if (po.tensor_id >= event_table.size()) continue;
        output_events.push_back(event_table[po.tensor_id]);
    }

    // 9. Move plan-output buffers back to the caller. Inputs and
    //    intermediates live and die inside the table — they get
    //    cleaned up when `buffer_table` goes out of scope.
    for (std::size_t i = 0; i < plan.outputs.size(); ++i) {
        const TensorId tid = plan.outputs[i].tensor_id;
        if (i >= outputs.size()) break;
        if (tid < buffer_table.size()) {
            outputs[i] = std::move(buffer_table[tid]);
        }
    }

    return Result<std::vector<Event>>{std::move(output_events)};
}

}  // namespace forge_rt
