// ForgeRT — JSON (de)serialization for ExecutionPlan (spec §10).
// All `to_json` / `from_json` overloads live in the `forge_rt`
// namespace so ADL picks them up at call sites inside the same
// namespace (and so callers don't have to write `forge_rt::to_json`).
#include "forge_rt/execution_plan.hpp"

#include <nlohmann/json.hpp>

#include <vector>

namespace forge_rt {

// Enum serialization note: DataType / MemorySpace / TargetKind are
// serialized as their underlying integer values. This keeps the JSON
// compact and preserves the ABI (§12) — the integer constants are
// stable, and reordering them is forbidden by the spec. If we ever
// need a human-readable form (e.g. for `forge plan inspect`) we can
// add a sidecar pretty-printer rather than changing the wire format.

// ---------------------------------------------------------------------------
// TensorDefinition
// ---------------------------------------------------------------------------
void to_json(nlohmann::json& j, const TensorDefinition& t) {
    j = nlohmann::json{
        {"id",            t.id},
        {"name",          t.name},
        {"dtype",         static_cast<uint8_t>(t.dtype)},
        {"shape",         t.shape},
        {"memory_space",  static_cast<uint8_t>(t.memory_space)},
    };
}

void from_json(const nlohmann::json& j, TensorDefinition& t) {
    j.at("id").get_to(t.id);
    j.at("name").get_to(t.name);
    t.dtype = static_cast<DataType>(j.at("dtype").get<uint8_t>());
    j.at("shape").get_to(t.shape);
    t.memory_space = static_cast<MemorySpace>(j.at("memory_space").get<uint8_t>());
}

// ---------------------------------------------------------------------------
// AllocationDefinition
// ---------------------------------------------------------------------------
void to_json(nlohmann::json& j, const AllocationDefinition& a) {
    j = nlohmann::json{
        {"tensor_id",    a.tensor_id},
        {"byte_offset",  a.byte_offset},
        {"byte_size",    a.byte_size},
        {"alignment",    a.alignment},
        {"memory_space", static_cast<uint8_t>(a.memory_space)},
    };
}

void from_json(const nlohmann::json& j, AllocationDefinition& a) {
    j.at("tensor_id").get_to(a.tensor_id);
    j.at("byte_offset").get_to(a.byte_offset);
    j.at("byte_size").get_to(a.byte_size);
    j.at("alignment").get_to(a.alignment);
    a.memory_space = static_cast<MemorySpace>(j.at("memory_space").get<uint8_t>());
}

// ---------------------------------------------------------------------------
// LaunchParameters
// ---------------------------------------------------------------------------
void to_json(nlohmann::json& j, const LaunchParameters& l) {
    j = nlohmann::json{
        {"grid",                l.grid},
        {"block",               l.block},
        {"shared_memory_bytes", l.shared_memory_bytes},
    };
}

void from_json(const nlohmann::json& j, LaunchParameters& l) {
    j.at("grid").get_to(l.grid);
    j.at("block").get_to(l.block);
    j.at("shared_memory_bytes").get_to(l.shared_memory_bytes);
}

// ---------------------------------------------------------------------------
// ExecutableNode
// ---------------------------------------------------------------------------
void to_json(nlohmann::json& j, const ExecutableNode& n) {
    j = nlohmann::json{
        {"id",           n.id},
        {"device",       n.device},
        {"artifact",     n.artifact},
        {"inputs",       n.inputs},
        {"outputs",      n.outputs},
        {"dependencies", n.dependencies},
        {"launch",       n.launch},
    };
}

void from_json(const nlohmann::json& j, ExecutableNode& n) {
    j.at("id").get_to(n.id);
    j.at("device").get_to(n.device);
    j.at("artifact").get_to(n.artifact);
    j.at("inputs").get_to(n.inputs);
    j.at("outputs").get_to(n.outputs);
    j.at("dependencies").get_to(n.dependencies);
    j.at("launch").get_to(n.launch);
}

// ---------------------------------------------------------------------------
// PlanInput / PlanOutput
// ---------------------------------------------------------------------------
void to_json(nlohmann::json& j, const PlanInput& i) {
    j = nlohmann::json{
        {"tensor_id",     i.tensor_id},
        {"external_name", i.external_name},
    };
}

void from_json(const nlohmann::json& j, PlanInput& i) {
    j.at("tensor_id").get_to(i.tensor_id);
    j.at("external_name").get_to(i.external_name);
}

void to_json(nlohmann::json& j, const PlanOutput& o) {
    j = nlohmann::json{
        {"tensor_id",     o.tensor_id},
        {"external_name", o.external_name},
    };
}

void from_json(const nlohmann::json& j, PlanOutput& o) {
    j.at("tensor_id").get_to(o.tensor_id);
    j.at("external_name").get_to(o.external_name);
}

// ---------------------------------------------------------------------------
// ExecutionPlan — top-level.
// Note: no version / checksum fields yet. Task 14 (compilation cache)
// introduces a hierarchical key; we'll add a `version` constant there
// rather than in this Phase 1 payload. See `ExecutionPlan` comment.
// ---------------------------------------------------------------------------
void to_json(nlohmann::json& j, const ExecutionPlan& p) {
    j = nlohmann::json{
        {"name",        p.name},
        {"tensors",     p.tensors},
        {"allocations", p.allocations},
        {"nodes",       p.nodes},
        {"inputs",      p.inputs},
        {"outputs",     p.outputs},
    };
}

void from_json(const nlohmann::json& j, ExecutionPlan& p) {
    j.at("name").get_to(p.name);
    j.at("tensors").get_to(p.tensors);
    j.at("allocations").get_to(p.allocations);
    j.at("nodes").get_to(p.nodes);
    j.at("inputs").get_to(p.inputs);
    j.at("outputs").get_to(p.outputs);
}

}  // namespace forge_rt
