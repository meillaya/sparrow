#pragma once

// ForgeRT C++ ABI — ExecutionPlan (spec §10).
// A static, topologically-ordered description of one runnable graph:
// tensors, allocations, executable nodes (one per launch), and the
// PlanInput/PlanOutput maps that say "this is the user's `x`, this is
// the user's `y`". The scheduler (Task 12) consumes an ExecutionPlan
// and dispatches each node to the matching DeviceProvider. JSON
// (de)serialization is via nlohmann_json ADL — see
// `runtime/lib/execution_plan.cpp` for the bodies; only the struct
// types live here so callers can pass them around without pulling in
// <nlohmann/json.hpp>.
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "abi_types.hpp"
#include "artifact_descriptor.hpp"
#include "result.hpp"

namespace forge_rt {

// Cheap, dense integer IDs used as map keys in the live buffer / event
// tables inside the scheduler. We don't use hashes or strings here
// because plans are built once and walked many times — integer keys
// keep the hot path branch-free.
using TensorId   = uint32_t;
using NodeId     = uint32_t;
using ArtifactId = uint32_t;
using DeviceId   = uint32_t;

// Describes one tensor view: dtype, shape, and which memory space the
// backing buffer lives in. Together with AllocationDefinition this is
// enough to materialize a BufferSpec at schedule time.
struct TensorDefinition {
    TensorId id = 0;
    std::string name;
    DataType dtype = DataType::Float32;
    std::vector<int64_t> shape;
    MemorySpace memory_space = MemorySpace::Host;
};

// A byte-offset / byte-size / alignment triple into a backing buffer.
// Phase 1 holds a single allocation per tensor; later versions can
// pack multiple tensors into one Buffer (the byte_offset is the
// intra-buffer slot). MemorySpace is duplicated from TensorDefinition
// so the scheduler doesn't have to round-trip through tensors when
// resolving providers.
struct AllocationDefinition {
    TensorId tensor_id = 0;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;
    uint64_t alignment = 0;
    MemorySpace memory_space = MemorySpace::Host;
};

// One device slot in a version 3 plan. Nodes reference the id, and the
// scheduler resolves it to a registered provider by target. Version 2
// plans have no device table; their `ExecutableNode::device` is a
// provider index.
struct DeviceDefinition {
    DeviceId id = 0;
    std::string name;
    TargetKind target = TargetKind::CPU;
    MemorySpace memory_space = MemorySpace::Host;
};

// One launch site: device, artifact, I/O lists, dependency edges, and
// launch geometry. `dependencies` is the set of NodeIds that must
// complete before this node fires — the scheduler materialises the
// deps as an Event span on `DeviceProvider::launch`.
struct ExecutableNode {
    NodeId id = 0;
    DeviceId device = 0;
    ArtifactId artifact = 0;
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    std::vector<NodeId> dependencies;
    LaunchParameters launch;
};

// Maps a PlanInput tensor (typically the user's "x") to an external
// buffer that the caller passes in via Scheduler::execute.
struct PlanInput {
    TensorId tensor_id = 0;
    std::string external_name;
};

// Maps a PlanOutput tensor (typically the user's "y") to a slot in the
// caller's outputs buffer vector.
struct PlanOutput {
    TensorId tensor_id = 0;
    std::string external_name;
};

// Top-level plan: a name, a schema version, the artifact table, and the
// live collections. Version 2 adds the artifact table. Version 3 adds
// the device table; version 1/2 plans remain readable and keep the
// provider-index meaning for `ExecutableNode::device`.
struct ExecutionPlan {
    static constexpr uint32_t kCurrentVersion = 3;

    uint32_t version = kCurrentVersion;
    std::string name;
    std::vector<ArtifactDescriptor> artifacts;
    std::vector<DeviceDefinition> devices;
    std::vector<TensorDefinition> tensors;
    std::vector<AllocationDefinition> allocations;
    std::vector<ExecutableNode> nodes;
    std::vector<PlanInput> inputs;
    std::vector<PlanOutput> outputs;
};

// ADL hooks for nlohmann_json. Forward-declared here; defined in
// runtime/lib/execution_plan.cpp. Putting the declarations in the
// `forge_rt` namespace makes nlohmann pick them up via ADL when
// callers do `to_json(j, plan)` / `from_json(j, plan)`.
void to_json(nlohmann::json& j, const TensorDefinition& t);
void from_json(const nlohmann::json& j, TensorDefinition& t);
void to_json(nlohmann::json& j, const AllocationDefinition& a);
void from_json(const nlohmann::json& j, AllocationDefinition& a);
void to_json(nlohmann::json& j, const DeviceDefinition& d);
void from_json(const nlohmann::json& j, DeviceDefinition& d);
void to_json(nlohmann::json& j, const ExecutableNode& n);
void from_json(const nlohmann::json& j, ExecutableNode& n);
void to_json(nlohmann::json& j, const PlanInput& i);
void from_json(const nlohmann::json& j, PlanInput& i);
void to_json(nlohmann::json& j, const PlanOutput& o);
void from_json(const nlohmann::json& j, PlanOutput& o);
void to_json(nlohmann::json& j, const ExecutionPlan& p);
void from_json(const nlohmann::json& j, ExecutionPlan& p);

// Validating reader. Returns an Error for an unsupported version, a
// negative or malformed id, or a node that names a missing artifact.
// `parse_plan_text` is the same check over a JSON string.
Result<ExecutionPlan> parse_plan_json(const nlohmann::json& j);
Result<ExecutionPlan> parse_plan_text(const std::string& text);

}  // namespace forge_rt
