// ForgeRT — topological scheduler (spec §10).
//
// Execution has four phases:
//   1. Validate the plan: ids, arity, artifact references, providers.
//   2. Compute the topological order once.
//   3. Plan liveness and buffer reuse in that order, then allocate every
//      intermediate tensor through the provider for its memory space.
//   4. Walk the order, build non-owning aliases for each node's inputs and
//      outputs, check output capacity, resolve the real artifact, and
//      launch.
//
// The scheduler never moves caller buffers. External inputs and outputs
// stay owned by the caller and are aliased into the live table; owned
// buffers exist only for intermediates. A failed run therefore leaves
// every caller buffer valid.
#include "forge_rt/scheduler.hpp"

#include "forge_rt/artifact_descriptor.hpp"
#include "forge_rt/memory_planner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace forge_rt {

Scheduler::Scheduler(std::vector<std::shared_ptr<DeviceProvider>> providers)
    : providers_(std::move(providers)) {}

namespace {

Error error(std::string message, int code) {
    return Error{std::move(message), code};
}

// Kahn's algorithm over the plan's node graph. Returns NodeIds in a
// deterministic order: ready nodes are seeded and appended in plan order.
Result<std::vector<NodeId>> topological_order(
    const ExecutionPlan& plan,
    const std::unordered_map<NodeId, std::size_t>& node_index) {
    std::unordered_map<NodeId, std::size_t> in_degree;
    std::unordered_map<NodeId, std::vector<NodeId>> successors;
    in_degree.reserve(node_index.size() * 2 + 1);
    successors.reserve(node_index.size() * 2 + 1);

    for (const ExecutableNode& node : plan.nodes) {
        in_degree.emplace(node.id, 0);
        successors.emplace(node.id, std::vector<NodeId>{});
    }
    for (const ExecutableNode& node : plan.nodes) {
        for (const NodeId dependency : node.dependencies) {
            if (!node_index.contains(dependency)) {
                return error("node " + std::to_string(node.id) +
                                 " depends on unknown node " +
                                 std::to_string(dependency),
                             -15);
            }
            successors[dependency].push_back(node.id);
            in_degree[node.id]++;
        }
    }

    std::deque<NodeId> ready;
    for (const ExecutableNode& node : plan.nodes) {
        if (in_degree[node.id] == 0) {
            ready.push_back(node.id);
        }
    }

    std::vector<NodeId> order;
    order.reserve(plan.nodes.size());
    while (!ready.empty()) {
        const NodeId current = ready.front();
        ready.pop_front();
        order.push_back(current);
        for (const NodeId successor : successors[current]) {
            if (in_degree[successor] == 0) {
                continue;
            }
            if (--in_degree[successor] == 0) {
                ready.push_back(successor);
            }
        }
    }

    if (order.size() != plan.nodes.size()) {
        return error("topological sort failed: cycle detected", -20);
    }
    return order;
}

void apply_tensor_spec(const TensorDefinition& tensor,
                       const AllocationDefinition& allocation,
                       Buffer& buffer) {
    BufferSpec& spec = buffer.spec();
    spec.descriptor.dtype = tensor.dtype;
    spec.descriptor.shape = tensor.shape;
    spec.descriptor.memory_space = allocation.memory_space;
    spec.descriptor.byte_offset = allocation.byte_offset;
    spec.alignment = allocation.alignment;
}

Result<void> ensure_capacity(Buffer& buffer, uint64_t required_bytes, TensorId tid) {
    if (buffer.spec().byte_size == 0) {
        buffer.spec().byte_size = required_bytes;
        return Result<void>{};
    }
    if (buffer.spec().byte_size < required_bytes) {
        return error("tensor " + std::to_string(tid) + " buffer is " +
                         std::to_string(buffer.spec().byte_size) +
                         " bytes, plan requires " + std::to_string(required_bytes),
                     -27);
    }
    return Result<void>{};
}

}  // namespace

Result<std::vector<Event>> Scheduler::execute(ExecutionPlan& plan,
                                              std::vector<Buffer>& inputs,
                                              std::vector<Buffer>& outputs) {
    if (plan.version == 0 || plan.version > ExecutionPlan::kCurrentVersion) {
        return error("unsupported plan version " + std::to_string(plan.version), -42);
    }
    if (plan.inputs.size() != inputs.size()) {
        return error("plan.inputs has " + std::to_string(plan.inputs.size()) +
                         " entries but caller passed " + std::to_string(inputs.size()),
                     -11);
    }
    if (plan.outputs.size() != outputs.size()) {
        return error("plan.outputs has " + std::to_string(plan.outputs.size()) +
                         " entries but caller passed " + std::to_string(outputs.size()),
                     -13);
    }

    // Dense slot per declared tensor. The table is sized by tensor count,
    // not by the largest id, so sparse or hostile ids cannot force a huge
    // allocation.
    std::unordered_map<TensorId, std::size_t> tensor_slot;
    tensor_slot.reserve(plan.tensors.size() * 2 + 1);
    for (std::size_t index = 0; index < plan.tensors.size(); ++index) {
        if (!tensor_slot.emplace(plan.tensors[index].id, index).second) {
            return error("duplicate tensor id " + std::to_string(plan.tensors[index].id),
                         -28);
        }
    }

    std::unordered_map<NodeId, std::size_t> node_index;
    node_index.reserve(plan.nodes.size() * 2 + 1);
    for (std::size_t index = 0; index < plan.nodes.size(); ++index) {
        if (!node_index.emplace(plan.nodes[index].id, index).second) {
            return error("duplicate node id " + std::to_string(plan.nodes[index].id), -28);
        }
        if (plan.artifacts.empty() ||
            plan.nodes[index].artifact >= plan.artifacts.size()) {
            return error("node " + std::to_string(plan.nodes[index].id) +
                             " references missing artifact " +
                             std::to_string(plan.nodes[index].artifact),
                         -18);
        }
    }

    const auto order_result = topological_order(plan, node_index);
    if (!order_result.ok()) {
        return order_result.error();
    }
    const std::vector<NodeId>& order = order_result.value();

    const auto planning_result = MemoryPlanner::plan(plan, order);
    if (!planning_result.ok()) {
        return error("memory planning failed: " + planning_result.error().message, -10);
    }

    std::unordered_map<TensorId, const TensorDefinition*> tensors;
    tensors.reserve(plan.tensors.size() * 2 + 1);
    for (const TensorDefinition& tensor : plan.tensors) {
        tensors.emplace(tensor.id, &tensor);
    }
    std::unordered_map<TensorId, const AllocationDefinition*> allocations;
    allocations.reserve(plan.allocations.size() * 2 + 1);
    for (const AllocationDefinition& allocation : plan.allocations) {
        allocations.emplace(allocation.tensor_id, &allocation);
    }

    std::unordered_set<TensorId> external;
    external.reserve(plan.inputs.size() + plan.outputs.size());
    for (const PlanInput& input : plan.inputs) {
        external.insert(input.tensor_id);
    }
    for (const PlanOutput& output : plan.outputs) {
        external.insert(output.tensor_id);
    }

    std::vector<Buffer> table(plan.tensors.size());
    for (std::size_t index = 0; index < plan.inputs.size(); ++index) {
        const TensorId tid = plan.inputs[index].tensor_id;
        const auto slot_it = tensor_slot.find(tid);
        if (slot_it == tensor_slot.end()) {
            return error("plan.inputs[" + std::to_string(index) +
                             "] references unknown tensor " + std::to_string(tid),
                         -12);
        }
        table[slot_it->second] =
            Buffer(inputs[index].data(), inputs[index].spec(), false);
    }
    for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
        const TensorId tid = plan.outputs[index].tensor_id;
        const auto slot_it = tensor_slot.find(tid);
        if (slot_it == tensor_slot.end()) {
            return error("plan.outputs[" + std::to_string(index) +
                             "] references unknown tensor " + std::to_string(tid),
                         -14);
        }
        table[slot_it->second] =
            Buffer(outputs[index].data(), outputs[index].spec(), false);
    }

    // Allocate every intermediate through the provider that owns its
    // memory space. External tensors keep the caller's storage.
    for (const TensorDefinition& tensor : plan.tensors) {
        if (external.contains(tensor.id)) {
            continue;
        }
        const auto allocation_it = allocations.find(tensor.id);
        if (allocation_it == allocations.end()) {
            return error("tensor " + std::to_string(tensor.id) +
                             " is missing from plan.allocations",
                         -17);
        }
        const AllocationDefinition& allocation = *allocation_it->second;
        std::shared_ptr<DeviceProvider> provider;
        for (const std::shared_ptr<DeviceProvider>& candidate : providers_) {
            if (candidate->memory_space() == allocation.memory_space) {
                provider = candidate;
                break;
            }
        }
        if (!provider) {
            return error("no provider for memory space " +
                             std::to_string(static_cast<int>(allocation.memory_space)),
                         -26);
        }
        BufferSpec spec;
        spec.descriptor.dtype = tensor.dtype;
        spec.descriptor.shape = tensor.shape;
        spec.descriptor.memory_space = allocation.memory_space;
        spec.descriptor.byte_offset = allocation.byte_offset;
        spec.byte_size = allocation.byte_size;
        spec.alignment = allocation.alignment;
        auto allocated = provider->allocate(spec);
        if (!allocated.ok()) {
            return error("allocation failed for tensor " +
                             std::to_string(tensor.id) + ": " +
                             allocated.error().message,
                         -26);
        }
        table[tensor_slot[tensor.id]] = std::move(allocated.value());
    }

    std::vector<Event> event_table(plan.tensors.size());
    for (const PlanInput& input : plan.inputs) {
        const auto slot_it = tensor_slot.find(input.tensor_id);
        if (slot_it != tensor_slot.end()) {
            event_table[slot_it->second].mark_completed();
        }
    }

    for (const NodeId node_id : order) {
        const ExecutableNode& node = plan.nodes[node_index[node_id]];
        if (node.device >= providers_.size()) {
            return error("node " + std::to_string(node.id) + " references device " +
                             std::to_string(node.device) + " but only " +
                             std::to_string(providers_.size()) + " providers are registered",
                         -22);
        }
        const std::shared_ptr<DeviceProvider>& provider = providers_[node.device];

        ArtifactDescriptor artifact = plan.artifacts[node.artifact];
        artifact.launch = node.launch;
        if (artifact.target != provider->target()) {
            return error("node " + std::to_string(node.id) + " artifact target " +
                             std::to_string(static_cast<int>(artifact.target)) +
                             " does not match provider target " +
                             std::to_string(static_cast<int>(provider->target())),
                         -22);
        }

        std::vector<Buffer> in_buffers;
        std::vector<Event> in_events;
        in_buffers.reserve(node.inputs.size());
        in_events.reserve(node.inputs.size());
        for (const TensorId tid : node.inputs) {
            const auto slot_it = tensor_slot.find(tid);
            if (slot_it == tensor_slot.end()) {
                return error("node " + std::to_string(node.id) +
                                 " references unknown input tensor " + std::to_string(tid),
                             -23);
            }
            const auto tensor_it = tensors.find(tid);
            const auto allocation_it = allocations.find(tid);
            if (tensor_it == tensors.end() || allocation_it == allocations.end()) {
                return error("tensor " + std::to_string(tid) +
                                 " is missing from the plan tables",
                             -17);
            }
            Buffer& buffer = table[slot_it->second];
            apply_tensor_spec(*tensor_it->second, *allocation_it->second, buffer);
            const auto capacity = ensure_capacity(
                buffer, allocation_it->second->byte_size, tid);
            if (!capacity.ok()) {
                return capacity.error();
            }
            in_buffers.emplace_back(buffer.data(), buffer.spec(), false);
            in_events.push_back(event_table[slot_it->second]);
        }

        std::vector<Buffer> out_buffers;
        std::vector<std::size_t> out_slots;
        out_buffers.reserve(node.outputs.size());
        out_slots.reserve(node.outputs.size());
        for (const TensorId tid : node.outputs) {
            const auto slot_it = tensor_slot.find(tid);
            if (slot_it == tensor_slot.end()) {
                return error("node " + std::to_string(node.id) +
                                 " references unknown output tensor " + std::to_string(tid),
                             -24);
            }
            const auto tensor_it = tensors.find(tid);
            const auto allocation_it = allocations.find(tid);
            if (tensor_it == tensors.end() || allocation_it == allocations.end()) {
                return error("tensor " + std::to_string(tid) +
                                 " is missing from the plan tables",
                             -17);
            }
            Buffer& buffer = table[slot_it->second];
            apply_tensor_spec(*tensor_it->second, *allocation_it->second, buffer);
            const auto capacity = ensure_capacity(
                buffer, allocation_it->second->byte_size, tid);
            if (!capacity.ok()) {
                return capacity.error();
            }
            out_buffers.emplace_back(buffer.data(), buffer.spec(), false);
            out_slots.push_back(slot_it->second);
        }

        auto launched = provider->launch(artifact, in_buffers, out_buffers, in_events);
        if (!launched.ok()) {
            return error("launch failed for node " + std::to_string(node.id) + ": " +
                             launched.error().message,
                         -30);
        }
        for (const std::size_t slot : out_slots) {
            event_table[slot] = launched.value();
        }
    }

    std::vector<Event> output_events;
    output_events.reserve(plan.outputs.size());
    for (const PlanOutput& output : plan.outputs) {
        const auto slot_it = tensor_slot.find(output.tensor_id);
        if (slot_it != tensor_slot.end()) {
            output_events.push_back(event_table[slot_it->second]);
        }
    }
    return output_events;
}

}  // namespace forge_rt
